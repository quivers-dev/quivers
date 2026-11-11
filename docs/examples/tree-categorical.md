# Tree-structured score tensor

## Overview

This program assembles log weights for four leaves of a binary tree and a per-verb/per-class score tensor. Its observation is `Normal(cell_score[0, 0], 0.5)`: it does not use a Categorical likelihood or select a class from the tree probabilities.

This example demonstrates both forms of the [`factor`](../guides/dsl-programs-and-lets.md#factor-expressions-assembling-indexed-tensors) expression. The pattern-match form builds the leaf-log-weight vector with a case table, and the multi-binder form builds a rank-2 score tensor by evaluating its body once per cell.

## QVR source

```qvr
# Tree-Structured Score Tensor
#
# A factor-expression example that assembles log weights for the
# leaves of a binary tree and then combines them with per-verb and
# per-class effects. The observation uses only cell_score[0, 0]
# under a Normal likelihood; the source does not define a
# Categorical outcome over the four leaves.
#
# Generative structure:
#
#   p_root         ~ Beta(1, 1)                    root split
#   p_left         ~ Beta(1, 1)                    left subtree split
#   p_right        ~ Beta(1, 1)                    right subtree split
#   sigma_v        ~ HalfNormal(1)                 per-verb scale
#   delta_v        ~ Normal(0, sigma_v)            per-verb effect
#   mu_k           ~ Normal(0, 1)                  per-class effect
#   y              ~ Normal(cell_score[0, 0], 0.5)
#
# Two forms of the factor expression appear: a pattern-match
# factor builds the tree-shaped leaf log-probability vector with
# a case table over Class (compiler-enforced label coverage of
# {0, ..., K-1}), and a multi-binder uniform factor builds the
# rank-2 score tensor by evaluating its body once per (Verb,
# Class) cell.

object Verb : FinSet 12
object Class : FinSet 4
object Resp : FinSet 200
object Val : Real 1

# Resp is the response plate: 200 rows of a single real response.
# Val is the program's codomain, the value space of what the
# program returns. It returns the per-verb effect vector delta,
# whose rows are single real numbers, so the codomain is Real 1
# and not the Resp index the observations are plated over.
program tree_categorical : Resp -> Val
    sample p_root <- Beta(concentration1=1.0, concentration0=1.0)
    sample p_left <- Beta(concentration1=1.0, concentration0=1.0)
    sample p_right <- Beta(concentration1=1.0, concentration0=1.0)

    let leaf_log = factor cls : Class in { 0 -> log(1.0 - p_root) + log(1.0 - p_left), 1 -> log(1.0 - p_root) + log(p_left), 2 -> log(p_root)       + log(1.0 - p_right), 3 -> log(p_root)       + log(p_right), }

    sample sigma_v <- HalfNormal(scale=1.0)
    sample delta : Verb <- Normal(loc=0.0, scale=sigma_v)
    sample mu : Class <- Normal(loc=0.0, scale=1.0)

    let cell_score = factor v : Verb, cls : Class in delta[v] + mu[cls] + leaf_log[cls]
    let cell0 = cell_score[0, 0]

    observe y : Resp <- Normal(loc=cell0, scale=0.5)
    return delta

export tree_categorical
```

## Understanding the model

A binary tree assigns a probability to a leaf by multiplying the branch
probabilities along its path. Taking logs turns that product into a sum.
The first `factor` expression writes out these different paths; the second
builds one score for every verb/class pair by combining the relevant entries.

Here `factor` assembles a tensor deterministically. The [observation likelihood](../tutorials/qvr/01-first-model.md#the-model)
reads only `cell_score[0, 0]`, so changing other cells while preserving that
entry leaves the likelihood unchanged. Learning those cells would require
observations that select them.

## Walkthrough

### What the four objects name

`Verb` and `Class` index the two axes of the constructed score tensor.
`Resp` indexes the rows scored by the Normal likelihood. The family makes
each observed value real; the response index does not determine its value
type. `Val : Real 1` describes the real entries of the returned `delta`
vector. Keeping these roles separate makes the declared shapes readable
without treating every finite set as a response space.

### Pattern-match factor: tree-shaped leaf probabilities

<!-- compile: false -->
```qvr
let leaf_log = factor cls : Class in {
    0 -> log(1.0 - p_root) + log(1.0 - p_left),
    1 -> log(1.0 - p_root) + log(p_left),
    2 -> log(p_root)       + log(1.0 - p_right),
    3 -> log(p_root)       + log(p_right),
}
```

The pattern-match form `factor cls : I in { 0 -> e_0, ..., n-1 -> e_{n-1} }` denotes a tensor of shape `(|I|, ...)` whose `i`-th cell is `e_i`. Here each leaf class is a structurally different product of internal-node log-probabilities, reflecting the geometry of a binary decision tree: classes 0 and 1 sit beneath the left child of the root (`1 - p_root` × left branch); classes 2 and 3 sit beneath the right (`p_root` × right branch). The compiler enforces label coverage of `{0, ..., |Class|-1}` exactly and rejects gaps, duplicates, or out-of-range labels at compile time.

This is the categorical representation of *structurally heterogeneous* indexed families: distributions over $\mathsf{Class}$ whose cells come from different upstream latents in different ways.

### Multi-binder uniform factor: the joint score tensor

<!-- compile: false -->
```qvr
let cell_score = factor v : Verb, cls : Class in
    delta[v] + mu[cls] + leaf_log[cls]
```

The multi-binder form `factor v_1 : I_1, ..., v_n : I_n in <body>` constructs a tensor of shape `(|I_1|, ..., |I_n|, *body_shape)` whose `(i_1, ..., i_n)`-th cell is the body evaluated with each binder `v_k := i_k`.

Here the body indexes into three previously-bound objects: the `Verb`-plate `delta`, the `Class`-plate `mu`, and the pattern-match factor `leaf_log` produced two steps earlier. Each `(v, cls)` cell evaluates to a different scalar; the resulting tensor lives on `Verb × Class` and carries the joint per-verb / per-class log-score.

The binder variables `v` and `cls` are integer-valued and visible only inside the body, mirroring the binder-localization of `let` in any functional language.

### Why factor and not a plate

A plate-bound draw `delta : Verb <- Normal(0.0, sigma_v)` and a factor expression `factor v : Verb in <body>` are not interchangeable. The plate draws an `|Verb|`-shape tensor of *independent* samples from the same kernel; the factor evaluates a *deterministic body* once per index and assembles the results into a tensor. The plate's family is exchangeable in its index; the factor's body is allowed to depend on the index in arbitrary structurally-different ways.

Use `factor` when cells need different expressions in upstream latents.
Here each leaf uses a different product of branch probabilities. A plate draw
instead applies one distribution family across an index, as in the gallery's
Normal and Dirichlet priors.

## Try it

> The short fits below demonstrate the API. Assess convergence with multiple chains and diagnostics before interpreting a posterior.


### Generating synthetic data

Pick true values for the three tree splits and for the per-verb and
per-class effects, assemble the observed cell score `cell_score[0, 0]`
from those same values by hand, and draw one response per row of the
`Resp` plate. Generating the data from the latents the program
samples is what makes the synthetic point self-consistent: a fit has
a ground truth to recover, and the reference oracle scores the joint
at a point the model itself could have produced.

```python
import math

import torch
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/tree_categorical.qvr")
model = prog.morphism

N_RESP, N_VERB = 200, 12

true_p_root = 0.7
true_p_left = 0.3
true_p_right = 0.6
true_sigma_v = 0.5
true_delta = true_sigma_v * torch.randn(N_VERB)
true_mu = torch.tensor([0.0, 0.5, -0.5, 1.0])

leaf_log_0 = math.log(1.0 - true_p_root) + math.log(1.0 - true_p_left)
cell0 = true_delta[0] + true_mu[0] + leaf_log_0
y = torch.distributions.Normal(cell0, 0.5).sample((N_RESP,))

observations = {"y": y}
x_in = torch.zeros(1, 1)
print("y batch shape:", tuple(y.shape))
```

### SVI fit

Reinitialize from the prior, then maximize the [`ELBO`](../api/inference/elbo.md#quivers.inference.objectives.ELBO) against the synthetic responses with an [`AutoNormalGuide`](../api/inference/guide.md#quivers.inference.guides.AutoNormalGuide) on the latent sites and [`SVI`](../api/inference/svi.md#svi) over [`Adam`](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html). Print the initial and final loss to confirm the guide is moving toward the posterior.

```python
import math

import torch
from quivers.dsl import load
from quivers.inference import AutoNormalGuide, ELBO, SVI

torch.manual_seed(0)
N_RESP, N_VERB = 200, 12
true_p_root, true_p_left = 0.7, 0.3
true_sigma_v = 0.5
true_delta = true_sigma_v * torch.randn(N_VERB)
true_mu = torch.tensor([0.0, 0.5, -0.5, 1.0])
leaf_log_0 = math.log(1.0 - true_p_root) + math.log(1.0 - true_p_left)
cell0 = true_delta[0] + true_mu[0] + leaf_log_0
obs = {"y": torch.distributions.Normal(cell0, 0.5).sample((N_RESP,))}

torch.manual_seed(1)
prog = load("docs/examples/source/tree_categorical.qvr")
model = prog.morphism
guide = AutoNormalGuide(model, observed_names=set(obs.keys()))
optim = torch.optim.Adam(
    list(model.parameters()) + list(guide.parameters()), lr=2e-2,
)
svi = SVI(model, guide, optim, ELBO())
svi_x = torch.zeros(1, 1)
loss0 = svi.step(svi_x, obs)
for _ in range(100):
    loss = svi.step(svi_x, obs)
print(f"ELBO loss: {loss0:.2f} -> {loss:.2f}")
```

### NUTS posterior

The tree-categorical program declares explicit `sample` priors for every latent (the three Beta splits, `sigma_v`, the per-verb `delta`, the per-class `mu`), so [`NUTSKernel`](../api/inference/mcmc.md#quivers.inference.mcmc.NUTSKernel) targets them directly without a parameter lift. For parameter-only models, the analogous step would route through [`bayesian_lift_parameters`](../api/inference/lifts.md#quivers.inference.lifts.bayesian_lift_parameters).

```python
import math

import torch
from quivers.dsl import load
from quivers.inference import MCMC, NUTSKernel

torch.manual_seed(0)
prog = load("docs/examples/source/tree_categorical.qvr")
model = prog.morphism

N_RESP, N_VERB = 200, 12
true_p_root, true_p_left = 0.7, 0.3
true_sigma_v = 0.5
true_delta = true_sigma_v * torch.randn(N_VERB)
true_mu = torch.tensor([0.0, 0.5, -0.5, 1.0])
leaf_log_0 = math.log(1.0 - true_p_root) + math.log(1.0 - true_p_left)
cell0 = true_delta[0] + true_mu[0] + leaf_log_0
obs = {"y": torch.distributions.Normal(cell0, 0.5).sample((N_RESP,))}

torch.manual_seed(2)
kernel = NUTSKernel(step_size=0.05, max_tree_depth=3, target_accept=0.8)
mc     = MCMC(kernel, num_warmup=10, num_samples=10, num_chains=1)
result = mc.run(model, torch.zeros(1, 1), obs)

print("acceptance:", float(result.acceptance_rates.mean()))
print("divergences:", int(result.divergence_counts.sum()))
```


## Categorical perspective

Operationally, `factor` constructs a tensor by finite index binding, while `arr[i]` eliminates an index by lookup. The compiler implements these two operations directly, and nothing on this page turns on a stronger adjunction or Kan-extension reading of them.

## See also

- [DSL Guide: Factor expressions](../guides/dsl-programs-and-lets.md#factor-expressions-assembling-indexed-tensors)
- [Mixture Model](mixture-model.md), the exchangeable counterpart: a flat Dirichlet over `Component` rather than a tree-shaped construction.
