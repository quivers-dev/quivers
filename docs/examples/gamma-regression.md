# Gamma Regression

## Overview

A Bayesian regression for strictly positive component totals, fit with a [Gamma](https://en.wikipedia.org/wiki/Gamma_distribution) likelihood and a per-category log-shape predictor. The model uses K = 3 categories: each category's log-shape is a linear function of the predictor, and the response is the raw per-cell Gamma tally rather than its simplex projection. This is the unnormalized layer of a Dirichlet regression [@maier2014dirichletreg]: independent Gamma draws with shapes `(alpha_1, ..., alpha_K)` and unit rate normalize to a sample on the [K-simplex](https://en.wikipedia.org/wiki/Simplex) via the [Gamma / Dirichlet relationship](https://en.wikipedia.org/wiki/Dirichlet_distribution#Gamma_distribution), but here the program observes the Gamma layer directly and leaves the deterministic normalization to downstream code.

## QVR source

```qvr
# Gamma Regression for Component Totals
#
# A Bayesian regression for strictly positive component totals, fit
# with a Gamma likelihood and a per-category log-shape predictor.
# This is the unnormalized layer of a Dirichlet regression:
# independent Gamma draws with shapes (alpha_1, ..., alpha_K)
# and unit rate normalize to a sample on the K-simplex, but the
# observed quantity here is the raw Gamma tally.
#
# Generative structure:
#
#   beta_0_k ~ Normal(0, 2)                       per-category intercept
#   beta_1_k ~ Normal(0, 2)                       per-category slope
#   tally_{n,k} ~ Gamma(exp(beta_0_k + beta_1_k * x_n), 1)
#
# The K coefficient plates are gathered through cat_idx so the
# program scales to arbitrary K without rewriting the body.
# Per-row simplex normalization is a deterministic
# post-composition handled downstream when the user wants a
# Dirichlet response.
#
# Reference: [Maier 2014](https://epub.wu.ac.at/4077/).

object Item : FinSet 21
object Cat : FinSet 3
object Resp : FinSet 63
object Val : Real 1

program gamma_regression : Resp -> Val
    sample beta_0 : Cat <- Normal(loc=0.0, scale=2.0)
    sample beta_1 : Cat <- Normal(loc=0.0, scale=2.0)

    let b0 = beta_0[cat_idx]
    let b1 = beta_1[cat_idx]
    let alpha = exp(b0 + b1 * x)

    observe tally : Resp <- Gamma(concentration=alpha, rate=1.0)
    return beta_1

export gamma_regression
```

## Understanding the model

A Gamma observation is a strictly positive amount, such as a duration or
component total. With unit rate, its shape `alpha` is both its mean and its
variance; exponentiating the [predictor](../tutorials/qvr/02-glms.md#poisson-regression) keeps `alpha` positive. The category
lookup selects which coefficients govern each flattened response cell.

Dividing a set of independent unit-rate Gamma draws by their sum produces a
Dirichlet-distributed vector of proportions. But observing the totals and
observing only their normalized proportions are different experiments. The
current likelihood scores the totals. Data containing only proportions would
instead need the Dirichlet density, which integrates out the unobserved total.

## Walkthrough

Per-category coefficient plates `beta_0 : Cat` and `beta_1 : Cat` carry one coefficient per category. The per-cell log-shape `b0 + b1 * x` is mapped through the exponential to give the positive Gamma shape `alpha`. The observed `tally : Resp <- Gamma(alpha, 1.0)` lives on the per-cell positive reals, indexed by the flattened `(Item, Cat)` axis. Plate-gather `beta_0[cat_idx]` selects each cell's category coefficient from the per-category plate.

To turn this into a Dirichlet regression on the K-simplex, normalize the tallies per Item: `y_{n, k} = tally_{n, k} / sum_j tally_{n, j}`. That projection is a deterministic post-composition on the Gamma response, applied outside the program because the QVR `Resp` axis is the flattened `(Item, Cat)` index and does not expose the K-axis as a named dimension to a `let`-level reducer.

The program returns `beta_1`, a `Cat`-indexed plate of real scalars, so the declared codomain is `Val : Real 1`: the per-row value space of the returned coefficients. `Resp` names the plate extent of the tallies and appears in the signature only as the domain.

## Try it

> The short fits below demonstrate the API. Assess convergence with multiple chains and diagnostics before interpreting a posterior.


### Generating synthetic data

```python
import torch
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/gamma_regression.qvr")
model = prog.morphism

N, K = 21, 3
NK = N * K
cat_idx = torch.arange(K).repeat(N)
x = torch.randn(NK)

true_beta_0 = torch.tensor([1.0, 0.5, -0.5])
true_beta_1 = torch.tensor([0.8, -1.0, 0.3])
alpha_true = torch.exp(true_beta_0[cat_idx] + true_beta_1[cat_idx] * x)
tally = torch.distributions.Gamma(alpha_true, 1.0).sample()

observations = {"x": x, "tally": tally, "cat_idx": cat_idx}
x_in = torch.zeros(1, 1)
```

### SVI fit

```python
from quivers.inference import AutoNormalGuide, ELBO, SVI

oracle_nll = float(
    -torch.distributions.Gamma(alpha_true, 1.0).log_prob(tally).mean()
)

torch.manual_seed(1)
guide = AutoNormalGuide(model, observed_names={"x", "tally", "cat_idx"})
optim = torch.optim.Adam(
    list(model.parameters()) + list(guide.parameters()), lr=5e-2,
)
svi = SVI(model, guide, optim, ELBO(num_particles=1))

losses = []
for _ in range(300):
    losses.append(svi.step(x_in, observations))

print(f"initial loss: {losses[0]:.2f}")
print(f"final loss:   {losses[-1]:.2f}")
print(f"oracle NLL:   {oracle_nll:.2f}")
```

### NUTS posterior

```python
from quivers.inference import MCMC, NUTSKernel

# The plate's size is declared in the source, so NUTS conditions on
# the whole data set, as one run of the program.
obs_mcmc = observations
x_in_mcmc = torch.zeros(1, 1)

torch.manual_seed(2)
kernel = NUTSKernel(step_size=0.05, max_tree_depth=3, target_accept=0.8)
mc = MCMC(kernel, num_warmup=20, num_samples=20, num_chains=1)
result = mc.run(model, x_in_mcmc, obs_mcmc)

print(f"acceptance:  {float(result.acceptance_rates.mean()):.2f}")
print(f"divergences: {int(result.divergence_counts.sum())}")
```


## Categorical perspective

The model factors as a K-fold tensor product of [`Gamma(alpha_k, 1)`](https://en.wikipedia.org/wiki/Gamma_distribution) kernels, one per category, with shape parameters driven by a per-category linear predictor through the exponential link. Pushing the joint Gamma kernel through the deterministic normalizing map `t |-> t / sum t` lands on the K-simplex as the [Dirichlet](https://en.wikipedia.org/wiki/Dirichlet_distribution) kernel, recovering Dirichlet regression as a post-composition. Working in the unnormalized Gamma layer keeps the per-category log-shapes linear in the predictor; the simplex projection would be a separate deterministic transformation supplied by the caller.


## Sources and further reading

- *DirichletReg: Dirichlet Regression for Compositional Data in R* [@maier2014dirichletreg].
