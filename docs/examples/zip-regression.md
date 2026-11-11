# Multi-output zero-inflated Poisson regression

## Overview

This program implements zero-inflated Poisson regression [@lambert1992zip]. Each output dimension carries logits for a Poisson-active gate and coefficients for the Poisson rate. At a `marginalize` head, the reference compiler enumerates the two endpoints of a Bernoulli-family prior, so `ContinuousBernoulli(pi_z)` contributes weights `(1 - pi_z, pi_z)` for the `Poisson(0)` and `Poisson(rate)` branches. The per-row likelihood is thus the exact two-component ZIP marginal.

## QVR source

```qvr
# Multi-Output Zero-Inflated Poisson Regression
#
# A multi-output zero-inflated Poisson regression. ZIP models
# count data with an excess of structural zeros relative to a
# plain Poisson likelihood: each cell is a two-component
# mixture of a point mass at zero and Poisson(rate). Each
# output dimension carries its own zero-inflation logits and
# rate coefficients.
#
# Generative structure:
#
#   alpha_zero_d, beta_zero_d ~ Normal(0, 5)
#   alpha_rate_d, beta_rate_d ~ Normal(0, 5)
#   pi_{n, d}                  = sigmoid(alpha_zero_d + beta_zero_d * x_n)
#   rate_{n, d}                = exp(alpha_rate_d + beta_rate_d * x_n)
#   z_{n, d}                   in {0, 1}, weights (1-pi, pi)
#   y_{n, d}                   ~ Poisson(z_{n, d} * rate_{n, d})
#
# At an ungrouped marginalize head, the QVR reference compiler
# interprets Bernoulli-family priors, including
# ContinuousBernoulli, on the finite endpoints {0, 1}. It scores
# both gated Poisson branches and combines them by logsumexp.
# Thus the block denotes the exact ZIP likelihood even though a
# forward trace may draw a continuous relaxation value for z.
#
# Reference: [Lambert 1992](https://doi.org/10.2307/1269547).

object Item : FinSet 200
object Out : FinSet 2
object Resp : FinSet 400
object Val : Real 1

program zip_regression : Resp -> Val
    sample alpha_zero : Out <- Normal(loc=0.0, scale=5.0)
    sample beta_zero : Out <- Normal(loc=0.0, scale=5.0)
    sample alpha_rate : Out <- Normal(loc=0.0, scale=5.0)
    sample beta_rate : Out <- Normal(loc=0.0, scale=5.0)

    let az = alpha_zero[out_idx]
    let bz = beta_zero[out_idx]
    let ar = alpha_rate[out_idx]
    let br = beta_rate[out_idx]
    let pi_z = sigmoid(az + bz * x)
    let rate = exp(ar + br * x)

    marginalize z : Resp <- ContinuousBernoulli(probs=pi_z)
        let gated_rate = z * rate
        observe y : Resp <- Poisson(rate=gated_rate)

    return beta_rate

export zip_regression
```

## Understanding the model

There are [two ways to observe zero](https://mc-stan.org/docs/stan-users-guide/finite-mixtures.html#zero-inflation):
the row belongs to the always-zero branch, or it belongs to the Poisson-active
branch and that draw happens to be zero [@stanFiniteMixtures]. With active
probability `pi` and rate `lambda`, the zero probability is
`(1 - pi) + pi * exp(-lambda)`. For a positive count, only the active branch
contributes to the likelihood.

The gate probability here is the probability of being active, despite the
`alpha_zero` and `beta_zero` names. The [reference marginalization](../tutorials/qvr/04-marginalize.md#a-two-component-gaussian-mixture) enumerates
both endpoints exactly for scoring, even though a forward trace can contain
a relaxed draw. The zero probability used to score observations is the
integrated value `(1 - pi) + pi * exp(-lambda)` above.

## Walkthrough

Per-output coefficient plates `alpha_zero`, `beta_zero` carry the [logit](https://en.wikipedia.org/wiki/Logit)-link Poisson-active probability `pi_{n,d}`, while `alpha_rate`, `beta_rate` carry the log-link rate. The `marginalize z : Resp` annotation gives each response row its own gate. The body is evaluated at `z = 0` and `z = 1`, then reduced by log-sum-exp with log weights `log(1 - pi_z)` and `log(pi_z)`. A forward trace may still record a relaxed draw for `z`, but that draw is score-suppressed; the integrated factor carries the joint density.

The program returns `beta_rate`, an `Out`-indexed plate of real scalars, so the declared codomain is `Val : Real 1`: the per-row value space of the returned coefficients. `Resp` names the flattened `(Item, Out)` plate extent and appears in the signature only as the domain.

## Try it

> The short fits below demonstrate the API. Assess convergence with multiple chains and diagnostics before interpreting a posterior.

### Generating synthetic data

```python
import torch
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/zip_regression.qvr")
model = prog.morphism

N, D = 200, 2
ND = N * D
out_idx = torch.arange(D).repeat(N)
x = torch.randn(ND)

true_alpha_zero = torch.tensor([0.5, 1.0])
true_beta_zero = torch.tensor([0.3, -0.2])
true_alpha_rate = torch.tensor([0.5, 1.0])
true_beta_rate = torch.tensor([1.0, -0.5])
pi_true = torch.sigmoid(true_alpha_zero[out_idx] + true_beta_zero[out_idx] * x)
rate_true = torch.exp(true_alpha_rate[out_idx] + true_beta_rate[out_idx] * x)
z_struct = torch.bernoulli(pi_true)
y = z_struct * torch.poisson(rate_true)

observations = {"x": x, "y": y, "out_idx": out_idx}
x_in = torch.zeros(1, 1)
```

### SVI fit

```python
from quivers.inference import AutoNormalGuide, ELBO, SVI

log_p_y0 = torch.log((1 - pi_true) + pi_true * torch.exp(-rate_true))
log_p_yk = torch.log(pi_true) + torch.distributions.Poisson(rate_true).log_prob(y)
oracle_nll = float(-torch.where(y == 0, log_p_y0, log_p_yk).mean())

torch.manual_seed(1)
guide = AutoNormalGuide(model, observed_names={"x", "y", "out_idx"})
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

torch.manual_seed(2)
kernel = NUTSKernel(step_size=0.05, max_tree_depth=3, target_accept=0.8)
mc = MCMC(kernel, num_warmup=20, num_samples=20, num_chains=1)
result = mc.run(model, x_in, observations)

print(f"acceptance:  {float(result.acceptance_rates.mean()):.2f}")
print(f"divergences: {int(result.divergence_counts.sum())}")
```

## Categorical perspective

The marginalized gate is a two-object coproduct: one branch maps to `Poisson(0)`, the other to `Poisson(rate)`, and log-sum-exp computes their weighted pushforward to the count space. The `ContinuousBernoulli` name controls the live forward draw, while the marginalization contract fixes the scored support to the two endpoints.

## Sources and further reading

- *Zero-Inflated Poisson Regression, With an Application to Defects in Manufacturing* [@lambert1992zip].
