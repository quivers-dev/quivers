# Transformer-shaped language model

## Overview

This model borrows the parallel-head and stacked-block shape of a [Transformer](https://papers.nips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html), then scores a token with a Categorical head. The `head` morphisms are independent MLP-parameterized Normal kernels. The source contains no query-key dot products, softmax over positions, causal mask, layer normalization, or additive residual connection. It is thus a wiring demonstration rather than a Transformer implementation or causal language model. [@vaswani2017attention]

## QVR source

```qvr
# Stacked Parallel-Branch Token Model
#
# A token model that borrows the parallel-head and stacked-block
# shape of a Transformer. Its heads are independent MLP-Normal
# kernels. There are no query-key products, softmax over
# positions, causal mask, layer normalization, or additive
# residual connections, so this is a wiring example rather than
# a Transformer implementation.
#
# Generative structure:
#
#   h_heads  ~ fan(head)(h) >> attn_proj           four parallel branches
#   h_res    ~ residual_attn(h_heads)              learned stochastic map
#   h_ff     ~ ff_up(h_res) >> ff_down             feed-forward block
#   h        ~ residual_ff(h_ff)                   feed-forward residual
#   next_t   ~ Categorical(logits=lm_head(h))      next-token target
#
# stack(layer, 2) composes two independently parameterized copies
# of the branch plus feed-forward block. The morphism weights are
# ordinary learned parameters unless an inference lift supplies
# priors over them.
#
# Resp is the plate: it indexes the 32 scored rows, one
# next-token target per context. Token is the vocabulary, so it
# is the value space of what lm_head draws and of what the
# program returns.
#
# Reference: [Vaswani et al. 2017](https://papers.nips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html).

object Token : FinSet 32
object Resp : FinSet 32
object Latent : Real 16
object HeadOut : Real 4
object FFHidden : Real 32

morphism tok_embed : Token -> Latent [role=embed]
morphism head : Latent -> HeadOut [replicate=4, param_source=mlp] ~ Normal
morphism attn_proj : Latent -> Latent [param_source=mlp] ~ Normal
morphism ff_up : Latent -> FFHidden [param_source=mlp] ~ Normal
morphism ff_down : FFHidden -> Latent [param_source=mlp] ~ Normal
morphism residual_attn, residual_ff : Latent -> Latent [param_source=mlp] ~ Normal
morphism lm_head : Latent -> Token ~ Categorical

define layer = fan(head) >> attn_proj >> residual_attn >> ff_up >> ff_down >> residual_ff
define backbone = tok_embed >> stack(layer, 2)

program transformer_lm : Token -> Token
    sample h <- backbone

    observe next_token : Resp <- lm_head(h)
    return next_token

export transformer_lm
```

## Understanding the model

Parallel branches answer several learned questions about the same input
vector, and their outputs are combined before the next layer. Stacking makes
that computation deeper, with new parameters at each layer. The final
Categorical kernel assigns probabilities to vocabulary entries, from which
the observed token receives a log-likelihood score.

The branches process one token vector per row. Their combined representation
parameterizes a distribution over the following token. Implementing attention
across positions would require a sequence axis, [query-key scores](tensor-contraction.md#understanding-the-model), and a weighted
sum over value vectors; those operations are absent from this source. Alammar’s [worked self-attention calculation](https://jalammar.github.io/illustrated-transformer/#self-attention-in-detail) shows those query, key, and value operations [@alammar2018transformer].

## Walkthrough

### Parallel heads

`morphism head : Latent -> HeadOut [replicate=4, param_source=mlp] ~ Normal` declares four independent kernels. [`fan(head)`](../guides/dsl-declarations.md#fan-out-diagonal-morphism) runs them in parallel on the same input and concatenates their four-dimensional outputs. This matches the dimensional wiring of four heads but does not compute attention.

### Layer block

<!-- compile: false -->
```qvr
define layer = fan(head) >> attn_proj >> residual_attn >> ff_up >> ff_down >> residual_ff
```

`attn_proj` mixes the parallel outputs back into `Latent`. `residual_attn` and `residual_ff` are sequential stochastic morphisms, not additions of a saved input, despite their names. `ff_up >> ff_down` is a two-stage MLP-shaped kernel composition.

### Deep stack

[`stack(layer, 2)`](../guides/dsl-declarations.md#stack-independent-multi-layer) creates two independent deep copies of `layer`, each with its own parameters (unlike [`repeat`](../guides/dsl-declarations.md#repeat-iterated-composition), which weight-ties the iterations). The full backbone is `tok_embed >> stack(layer, 2)`, mapping the input token sequence to a per-position `Latent` representation.

### Language-model head

The closing `morphism lm_head : Latent -> Token ~ Categorical` is a Kleisli morphism `Latent -> Token`; per position it produces a Categorical distribution over the thirty-two-symbol vocabulary, and the program's `observe next_token` step accumulates the per-position categorical log-likelihood against the supplied target tensor.

The two `FinSet` objects sit in different positions and mean different things. `Resp : FinSet 32` fills the observe step's index slot, so it is the plate: 32 scored rows. `Token : FinSet 32` fills `lm_head`'s codomain and the program's own codomain, so it is the value space: the 32 outcomes a draw ranges over, and the space the returned `next_token` lives in. That the two happen to have the same cardinality here is a coincidence of this example's sizing, not a shared role.

## Try it

> The short fits below demonstrate the API. Assess convergence with multiple chains and diagnostics before interpreting a posterior.


### Generating synthetic data

Fix the model's parameters to a chosen draw, then run one forward [`trace`](../api/inference/trace.md) so the latent representation `h` and the next-token targets generated from it are jointly consistent. `true_h` names the ground truth for the latent `h` site, and shipping it in the observations dict is what clamps it: an unclamped `h` is redrawn on every call, which leaves any reference joint non-deterministic. The corpus is 32 single-token contexts, one per element of the `Resp` plate, paired with 32 next-token targets. A one-position context does not test cross-position attention or causal masking: `fan(head)` inside `stack` folds a multi-position axis into the feature axis, so the composite as written scores one position per row.

```python
import torch
from quivers.dsl import load
from quivers.inference.trace import trace

torch.manual_seed(0)
prog = load("docs/examples/source/transformer_lm.qvr")
model = prog.morphism

for _, p in model.named_parameters():
    p.data.copy_(torch.randn_like(p) * 0.3)

rows, vocab = 32, 32
x_in = torch.randint(0, vocab, (rows,))
with torch.no_grad():
    forward = trace(model, x_in)
true_h = forward.sites["h"].value.detach()
next_token = forward.sites["next_token"].value.detach()

observations = {"next_token": next_token, "h": true_h}
print("x_in:", tuple(x_in.shape))
print("true_h:", tuple(true_h.shape))
print("next_token:", tuple(next_token.shape))
```

### SVI fit

Fit the illustrative model to synthetic observations with [`AutoNormalGuide`](../api/inference/guide.md), [`ELBO`](../api/inference/elbo.md), and [`SVI`](../api/inference/svi.md). Each step estimates the objective using random samples, so losses can fluctuate. Compare losses over several steps and evaluate the fitted next-token probabilities on a separate batch of synthetic contexts.

```python
import torch
from quivers.dsl import load
from quivers.inference import AutoNormalGuide, ELBO, SVI

torch.manual_seed(0)
prog = load("docs/examples/source/transformer_lm.qvr")
model = prog.morphism

for _, p in model.named_parameters():
    p.data.copy_(torch.randn_like(p) * 0.3)
rows, vocab = 32, 32
contexts = torch.randint(0, vocab, (rows,))
targets = model.rsample(contexts)
observations = {"next_token": targets}

torch.manual_seed(1)
for _, p in model.named_parameters():
    p.data.copy_(torch.randn_like(p) * 0.3)

guide = AutoNormalGuide(model, observed_names={"next_token"})
optim = torch.optim.Adam(
    list(model.parameters()) + list(guide.parameters()), lr=5e-2,
)
svi = SVI(model, guide, optim, ELBO(num_particles=1))

losses = [svi.step(contexts, observations)]
for _ in range(8):
    losses.append(svi.step(contexts, observations))

print(f"initial loss: {losses[0]:.2f}")
print(f"final loss:   {losses[-1]:.2f}")
```

### HMC posterior

The lifted Bayesian model treats both the parameters $\theta$ and the per-position latent $h$ as random variables: $p(\theta, h \mid x, y) \propto p(\theta) \, p(h \mid x, \theta) \, p(y \mid h, \theta)$. [`bayesian_lift_parameters`](../api/inference/lifts.md#quivers.inference.lifts.bayesian_lift_parameters) assigns Normal priors to every `nn.Parameter` and exposes the intermediate `h` site through `additional_latents`. The lifted log density is deterministic given the full $(\theta, h)$ state. Because the transformer's `log_joint` walks every step in the stack, NUTS tree expansion is expensive at this dimension; the example uses [`HMCKernel`](../api/inference/mcmc.md#quivers.inference.mcmc.HMCKernel) with one leapfrog step while retaining the same target distribution.

```python
import torch
from quivers.dsl import load
from quivers.inference import MCMC, HMCKernel, bayesian_lift_parameters

torch.manual_seed(0)
prog = load("docs/examples/source/transformer_lm.qvr")
model = prog.morphism
for _, p in model.named_parameters():
    p.data.copy_(torch.randn_like(p) * 0.3)
rows, vocab = 32, 32
contexts = torch.randint(0, vocab, (rows,))
targets = model.rsample(contexts)
observations = {"next_token": targets}

h_shape = tuple(model._step_h.rsample(contexts).shape)
lifted, lx, lobs = bayesian_lift_parameters(
    model, contexts, observations,
    prior_scale=1.0,
    additional_latents={"h": h_shape},
)
# The full transformer log_joint is expensive; use fixed-step HMC
# with one leapfrog step per sample to keep the run tractable.
# NUTS with the same target produces the same chain mathematically
# at much higher cost.
kernel = HMCKernel(step_size=0.001, num_steps=1, target_accept=0.6)
mc     = MCMC(kernel, num_warmup=3, num_samples=3, num_chains=1)
result = mc.run(lifted, lx, lobs)

print(f"acceptance:  {float(result.acceptance_rates.mean()):.2f}")
print(f"divergences: {int(result.divergence_counts.sum())}")
```


## Categorical perspective

The model composes replicated stochastic branches, a projection, and two-stage feed-forward kernels. [`stack`](../guides/dsl-declarations.md#stack-independent-multi-layer) makes independently parameterized layers, while [`fan`](../guides/dsl-declarations.md#fan-out-diagonal-morphism) copies the input across parallel branches. Neither combinator by itself implements self-attention.


## Sources and further reading

- *Attention is All you Need* [@vaswani2017attention].
