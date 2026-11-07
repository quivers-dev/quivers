# LSTM-shaped language model

## Overview

This stochastic recurrent cell borrows the gates of an [LSTM](https://doi.org/10.1162/neco.1997.9.8.1735), but it threads only `h_t`. Its local update uses `f_t * h_{t-1}` rather than a separately propagated `f_t * c_{t-1}`. The equations below specify the resulting stochastic recurrence. [@hochreiter1997lstm]

## QVR source

```qvr
# LSTM-Shaped Stochastic Language Model
#
# A gated stochastic recurrence wrapped in scan and used as a
# causal language model. It borrows three LSTM gates and a cell
# candidate, but scan threads only h_t: the local update uses
# h_{t-1} where a canonical LSTM carries a separate c_{t-1}.
# The tanh readout is realized through sigmoid.
#
# Generative structure:
#
#   i_t      ~ LogitNormal(gate_i(x_t, h_{t-1}))  input gate
#   f_t      ~ LogitNormal(gate_f(x_t, h_{t-1}))  forget gate
#   o_t      ~ LogitNormal(gate_o(x_t, h_{t-1}))  output gate
#   g_t      ~ Normal(cell_cand(x_t, h_{t-1}))    cell candidate
#   h_t      = o_t * tanh(f_t * h_{t-1} + i_t * g_t) folded-state update
#   next_t   ~ Categorical(logits=lm_head(h_t))   next-token target
#
# Resp is the plate: it indexes the 32 scored rows of the corpus,
# one next-token target per context. Token is the vocabulary, so it
# is the value space of what lm_head draws and of what the program
# returns.
#
# The intermediate c_new is local to one call and is not carried
# to the next position. This is thus an LSTM-shaped example,
# not an implementation of the canonical architecture.
#
# Reference: [Hochreiter and Schmidhuber 1997](https://doi.org/10.1162/neco.1997.9.8.1735).

object Token : FinSet 256
object Resp : FinSet 32
object Embedded : Real 64
object Hidden : Real 128

morphism tok_embed : Token -> Embedded [role=embed]
morphism gate_i, gate_f, gate_o : Embedded * Hidden -> Hidden ~ LogitNormal
morphism cell_cand : Embedded * Hidden -> Hidden ~ Normal
morphism lm_head : Hidden -> Token ~ Categorical

program lstm_cell(x_t, h_prev) : Embedded * Hidden -> Hidden
    sample i_gate <- gate_i(x_t, h_prev)
    sample f_gate <- gate_f(x_t, h_prev)
    sample o_gate <- gate_o(x_t, h_prev)
    sample g_cand <- cell_cand(x_t, h_prev)

    let c_new = f_gate * h_prev + i_gate * g_cand
    let two_c = 2.0 * c_new
    let sig_2c = sigmoid(two_c)
    let tanh_c = 2.0 * sig_2c - 1.0
    let h_new = o_gate * tanh_c
    return h_new

define backbone = tok_embed >> scan(lstm_cell)

program lstm_lm : Token -> Token
    sample h <- backbone

    observe next_token : Resp <- lm_head(h)
    return next_token

export lstm_lm
```

## Understanding the model

The input gate determines how much candidate information enters the local
update; the forget gate weights the previous hidden state; the output gate
weights the transformed update sent onward. Each gate lies between zero and
one, so multiplication acts as a learned proportion. Here those proportions
are sampled from LogitNormal kernels rather than fixed by a sigmoid alone.

A canonical LSTM carries both a [cell state and a hidden state](https://colah.github.io/posts/2015-08-Understanding-LSTMs/#the-core-idea-behind-lstms)
through time [@olah2015lstm]. This source passes only the hidden state to the
next step: `c_new` contributes to the current output and is then discarded.
Its terminal hidden state predicts one next token.

## Walkthrough

### Cell equations

The parametric program samples three gates and a candidate. `LogitNormal` constrains gate draws to $(0, 1)$, and the candidate is a Normal kernel. Inside the program body:

| Step | DSL | Meaning |
|---|---|---|
| input gate | `i_gate <- gate_i(x_t, h_prev)` | $i_t \sim \mathrm{LogitNormal}(\mu_i(x_t,h_{t-1}), \sigma_i(x_t,h_{t-1}))$ |
| forget gate | `f_gate <- gate_f(x_t, h_prev)` | $f_t = \sigma(W_f [x_t, h_{t-1}])$ |
| output gate | `o_gate <- gate_o(x_t, h_prev)` | $o_t \sim \mathrm{LogitNormal}(\mu_o(x_t,h_{t-1}), \sigma_o(x_t,h_{t-1}))$ |
| candidate | `g_cand <- cell_cand(x_t, h_prev)` | $g_t \sim \mathcal{N}(\mu_g(x_t,h_{t-1}), \sigma_g(x_t,h_{t-1}))$ |
| cell update | `let c_new = f_gate * h_prev + i_gate * g_cand` | $c_t = f_t \odot h_{t-1} + i_t \odot g_t$ |
| hidden | `let h_new = o_gate * tanh_c` | $h_t = o_t \odot \tanh(c_t)$ |

`tanh` is realized from [`sigmoid`](../guides/dsl-programs-and-lets.md#indexed-gather-in-let) via the identity $\tanh(x) = 2\,\sigma(2x) - 1$.

### State threading

`scan(lstm_cell)` is an iterated Kleisli composition along the sequence: the threaded state is the per-step hidden output $h_t$, which the Categorical `lm_head` reads at the terminal position to score the next token. The cell-state vector $c_t$ is computed inside the cell at every step from the threaded $h_{t-1}$ and used immediately to form $h_t$ via the output-gate / $\tanh$ post-composition. Because `scan` threads a single codomain, this presentation folds the canonical LSTM's separate $c$-channel into the local cell body: the long-term cell-state memory channel that a two-state LSTM exposes is not propagated across time steps here, and the recurrence reduces to $h_t = o_t \odot \tanh(f_t \odot h_{t-1} + i_t \odot g_t)$.

The two `FinSet` objects play different roles. `Resp : FinSet 32` sits in the observe step's index slot, so it is the plate: 32 scored rows, one next-token target per context. `Token : FinSet 256` sits in `lm_head`'s codomain and in the program's own codomain, so it is the value space the draw ranges over.

```mermaid
flowchart LR
    x_t["x_t"] --> gate_i["gate_i"]
    x_t["x_t"] --> gate_f["gate_f"]
    x_t["x_t"] --> gate_o["gate_o"]
    x_t["x_t"] --> cell_cand["cell_cand"]
    h_prev["h_prev"] --> gate_i["gate_i"]
    h_prev["h_prev"] --> gate_f["gate_f"]
    h_prev["h_prev"] --> gate_o["gate_o"]
    h_prev["h_prev"] --> cell_cand["cell_cand"]
    gate_f["gate_f"] --> c_new["c_new"]
    cell_cand["cell_cand"] --> c_new["c_new"]
    gate_i["gate_i"] --> c_new["c_new"]
    h_prev["h_prev"] --> c_new["c_new"]
    gate_o["gate_o"] --> h_new["h_new"]
    c_new["c_new"] --> h_new["h_new"]
    h_new["h_new"] --> scan["scan"]
    class x_t,h_prev qv-input
    class gate_i,gate_f,gate_o,cell_cand qv-checked
    class c_new,h_new qv-foundation
    class scan qv-output
```

## Try it

> The short fits below demonstrate the API. Assess convergence with multiple chains and diagnostics before interpreting a posterior.

### Generating synthetic data

Trace the fixed LSTM-shaped recurrence once to obtain its terminal `h` and target. Reuse `true_h` in `observations` so later likelihood evaluations score that same realization; otherwise each evaluation resamples `h`. Each `Resp` row contains an int64 prompt of length `seq_len` and one next-token label.

```python
import torch
from quivers.dsl import load
from quivers.inference.trace import trace

torch.manual_seed(0)
prog = load("docs/examples/source/lstm_lm.qvr")
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
true_h = forward.sites["h"].value.detach()
next_token = forward.sites["next_token"].value.detach()

x_in = prompts
observations = {"next_token": next_token, "h": true_h}
print("prompts:", prompts.shape, prompts.dtype)
print("true_h:", true_h.shape)
print("next_token:", next_token.shape, next_token.dtype)
```

### SVI fit

Reinitialize `tok_embed`, the three gate morphisms, `cell_cand`, and `lm_head`, then fit their latent parameters with an [`AutoNormalGuide`](../api/inference/guide.md) and [`SVI`](../api/inference/svi.md). The [`ELBO`](../api/inference/elbo.md) scores the Categorical `next_token` site for each prompt.

```python
import torch
from quivers.dsl import load
from quivers.inference import AutoNormalGuide, ELBO, SVI

torch.manual_seed(0)
prog = load("docs/examples/source/lstm_lm.qvr")
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

guide = AutoNormalGuide(model, observed_names={"next_token"})
optim = torch.optim.Adam(
    list(model.parameters()) + list(guide.parameters()), lr=5e-2,
)
svi = SVI(model, guide, optim, ELBO(num_particles=1))

losses = [svi.step(prompts, observations)]
for _ in range(30):
    losses.append(svi.step(prompts, observations))

print(f"initial loss: {losses[0]:.2f}")
print(f"final loss:   {losses[-1]:.2f}")
```

### NUTS posterior

The LSTM's four gates and cell candidate are kernel Bayesian morphisms whose weights live as `nn.Parameter`s inside the program. [`bayesian_lift_parameters`](../api/inference/lifts.md#quivers.inference.lifts.bayesian_lift_parameters) lifts those parameters into Normal-prior sample sites so [`NUTSKernel`](../api/inference/mcmc.md#quivers.inference.mcmc.NUTSKernel) has a continuous unconstrained state space. The likelihood scores the next-token target via the Categorical [`lm_head`](../api/continuous/families.md) applied to a forward sample of the hidden state.

```python
import torch
from quivers.dsl import load
from quivers.inference import MCMC, NUTSKernel, bayesian_lift_parameters

torch.manual_seed(0)
prog = load("docs/examples/source/lstm_lm.qvr")
model = prog.morphism
for _, p in model.named_parameters():
    p.data.copy_(torch.randn_like(p) * 0.3)
rows, seq_len, vocab = 32, 8, 256
prompts = torch.randint(0, vocab, (rows, seq_len))
targets = model.rsample(prompts)
observations = {"next_token": targets}

h_shape = tuple(model._step_h.rsample(prompts).shape)
lifted, lx, lobs = bayesian_lift_parameters(
    model, prompts, observations,
    prior_scale=1.0,
    additional_latents={"h": h_shape},
)
kernel = NUTSKernel(step_size=0.005, max_tree_depth=3, target_accept=0.8)
mc     = MCMC(kernel, num_warmup=10, num_samples=10, num_chains=1)
result = mc.run(lifted, lx, lobs)

print(f"acceptance:  {float(result.acceptance_rates.mean()):.2f}")
print(f"divergences: {int(result.divergence_counts.sum())}")
```

## Categorical perspective

The cell program denotes a Kleisli morphism $\mathrm{Embedded} \times \mathrm{Hidden} \to \mathcal{G}(\mathrm{Hidden})$ in the Kleisli category of the [Giry monad](https://doi.org/10.1007/BFb0092872); `scan(lstm_cell)` is its iterated composition over the sequence. The Categorical head closes the composite with a finite-set codomain, and `observe next_token` accumulates per-batch categorical log-likelihood through a [right Kan extension](https://ncatlab.org/nlab/show/Kan+extension). [@giry1982categorical]

## Sources and further reading

- *A categorical approach to probability theory* [@giry1982categorical].
- *Long Short-Term Memory* [@hochreiter1997lstm].
