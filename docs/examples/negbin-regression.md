# Multi-Output Negative Binomial Regression

## Overview

A multi-output [negative-binomial](https://en.wikipedia.org/wiki/Negative_binomial_distribution)
regression gives each output its own coefficients and count parameter.
The source uses PyTorch's `NegativeBinomial(total_count, probs)` convention.
Its intermediate variable `mu` is not the distribution mean under the
probability expression shown below; the walkthrough states that distinction
explicitly so the fitted coefficients have the correct interpretation.

## QVR source

```qvr
# Multi-Output Negative Binomial Regression
#
# Per-output coefficients and count parameters are gathered to response rows.
# The inline family follows PyTorch's NegativeBinomial convention:
# mean = total_count * probs / (1 - probs).
# With probs = disp / (disp + mu), the actual mean is disp**2 / mu.
# For the usual NB2 mean mu, use probs = mu / (disp + mu) instead.
# This source preserves the former probability expression.

object Item : FinSet 21
object Out : FinSet 3
object Resp : FinSet 63
object Val : Real 1

program negbin_regression : Resp -> Val
    sample beta_0 : Out <- Normal(loc=0.0, scale=5.0)
    sample beta_1 : Out <- Normal(loc=0.0, scale=5.0)
    sample dispersion : Out <- Gamma(concentration=2.0, rate=0.5)

    let b0 = beta_0[out_idx]
    let b1 = beta_1[out_idx]
    let disp = dispersion[out_idx]
    let eta = b0 + b1 * x
    let mu = exp(eta)
    let probs = disp / (disp + mu)

    observe y : Resp <- NegativeBinomial(total_count=disp, probs=probs)
    return beta_1

export negbin_regression
```

## Understanding the model

[Poisson variation](../tutorials/qvr/02-glms.md#poisson-regression) ties the variance of a count to its mean. A negative
binomial allows extra variation, as if an underlying Poisson rate itself
varied across observations. The distribution's count parameter and probability
parameter jointly set its mean; their exact convention matters.

In this source, `mu = exp(eta)` is an intermediate positive predictor. Under
the PyTorch convention used by the inline family, the displayed
`probs = disp / (disp + mu)` gives mean `disp**2 / mu`, not `mu`.
Thus increasing `eta` lowers this model's actual count mean at fixed `disp`.
The source and synthetic generator use that same convention; to specify the
usual NB2 mean `mu`, a model author would instead use
`probs = mu / (disp + mu)`.

## Walkthrough

Per-output coefficient and dispersion plates broadcast through `out_idx` gathers. The per-cell linear predictor `eta = b0 + b1 * x` is mapped through the log link `exp` to give the positive intermediate value `mu`. Under PyTorch's convention, `probs = dispersion / (dispersion + mu)` gives actual mean `m = dispersion**2 / mu` and variance `m + m**2 / dispersion`. To use `mu` as the NB2 mean, the probability expression would be `mu / (dispersion + mu)`. The Gamma prior on dispersion encodes a soft preference for finite overdispersion; per-output dispersion permits heterogeneous count regimes across the response axis.

The program returns `beta_1`, an `Out`-indexed plate of real scalars, so the declared codomain is `Val : Real 1`: the per-row value space of the returned coefficients. `Resp` names the plate extent of the response and appears in the signature only as the domain.

## Try it

> The short fits below demonstrate the API. Assess convergence with multiple chains and diagnostics before interpreting a posterior.


### Generating synthetic data

```python
import torch
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/negbin_regression.qvr")
model = prog.morphism

N, D = 21, 3
ND = N * D
out_idx = torch.arange(D).repeat(N)
x = torch.randn(ND)

true_beta_0 = torch.tensor([1.0, 0.5, 2.0])
true_beta_1 = torch.tensor([0.5, -0.3, 0.8])
true_dispersion = torch.tensor([5.0, 10.0, 3.0])
mu_true = torch.exp(true_beta_0[out_idx] + true_beta_1[out_idx] * x)
probs_true = true_dispersion[out_idx] / (true_dispersion[out_idx] + mu_true)
y = torch.distributions.NegativeBinomial(true_dispersion[out_idx], probs_true).sample()

observations = {"x": x, "y": y, "out_idx": out_idx}
x_in = torch.zeros(1, 1)
```

### SVI fit

```python
from quivers.inference import AutoNormalGuide, ELBO, SVI

oracle_nll = float(
    -torch.distributions.NegativeBinomial(true_dispersion[out_idx], probs_true)
    .log_prob(y)
    .mean()
)

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

The negative binomial is the Gamma-Poisson [mixture](https://en.wikipedia.org/wiki/Compound_probability_distribution): a `Poisson(rate)` kernel with `rate ~ Gamma(dispersion, mu / dispersion)` marginalizes, under PyTorch's parameter convention, to the source's `NegativeBinomial(dispersion, dispersion / (dispersion + mu))`. The Gamma rate is `mu / dispersion` because this source's count mean is `dispersion**2 / mu`. The model uses this closed-form marginal per cell.
