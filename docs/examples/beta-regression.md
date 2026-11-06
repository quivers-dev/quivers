# Multi-Output Beta Regression

## Overview

A multi-output beta regression [@ferrari2004beta] for response variables on the open unit interval. Each output dimension carries its own logit-link coefficients and precision; the per-cell mean is mapped through the [sigmoid](https://en.wikipedia.org/wiki/Logistic_function) link to the unit interval, and the [Beta](https://en.wikipedia.org/wiki/Beta_distribution) likelihood is reparameterized in mean / precision form.

## QVR source

```qvr
# Multi-Output Beta Regression
#
# A multi-output beta regression for response variables on the
# open unit interval. Each output dimension carries its own
# logit-link coefficients and precision; the per-cell mean is
# mapped through the sigmoid, and the Beta likelihood is
# parameterised in its mean / precision form.
#
# Generative structure:
#
#   beta_0_d ~ Normal(0, 5)                       per-output intercept
#   beta_1_d ~ Normal(0, 5)                       per-output slope
#   phi_d    ~ HalfCauchy(2.5)                    per-output precision
#   y_{n,d}  ~ Beta(mu * phi_d, (1 - mu) * phi_d) mu = sigmoid(eta)
#
# The mean / precision parameterisation gives mean mu and
# variance mu * (1 - mu) / (1 + phi); per-output precision
# permits heterogeneous dispersion across the response axis.
#
# Reference: [Ferrari and Cribari-Neto 2004](https://doi.org/10.1080/0266476042000214501).

object Item : FinSet 21
object Out : FinSet 3
object Resp : FinSet 63
object Val : Real 1

program beta_regression : Resp -> Val
    sample beta_0 : Out <- Normal(loc=0.0, scale=5.0)
    sample beta_1 : Out <- Normal(loc=0.0, scale=5.0)
    sample phi : Out <- HalfCauchy(scale=2.5)

    let b0 = beta_0[out_idx]
    let b1 = beta_1[out_idx]
    let p = phi[out_idx]
    let eta = b0 + b1 * x
    let mu = sigmoid(eta)
    let alpha = mu * p
    let beta_p = (1.0 - mu) * p

    observe y : Resp <- Beta(concentration1=alpha, concentration0=beta_p)
    return beta_1

export beta_regression
```

## Understanding the model

A fraction such as time spent on a task must stay between zero and one.
A [linear predictor](../tutorials/qvr/02-glms.md#logistic-regression) cannot ensure this, so `sigmoid` converts it to a valid
mean. With `mu = 0.5`, increasing `phi` concentrates outcomes near one-half;
with the same `phi`, changing `mu` moves their center. Exact zeros and ones
need a different observation model or an explicit boundary treatment.

For multiple outputs, [`out_idx`](../tutorials/qvr/09-indexed-data.md#5-transfer-the-pattern-to-statistical-data) tells each flattened response row which
coefficient and precision to use. If a row belongs to output 2, it gathers
`beta_0[2]`, `beta_1[2]`, and `phi[2]`. Posterior slopes describe changes on
the log-odds scale of the mean, so compare predicted fractions when the
scientific question is about differences on the original response scale.

## Walkthrough

Per-output coefficient plates `beta_0 : Out` and `beta_1 : Out` carry one coefficient per response dimension; per-output precision `phi : Out` permits heterogeneous dispersion across the response axis. The per-cell linear predictor `eta = b0 + b1 * x` is mapped to the unit interval via `mu = sigmoid(eta)`. The mean / precision form of the Beta is `Beta(mu * phi, (1 - mu) * phi)`, giving mean `mu` and variance `mu * (1 - mu) / (1 + phi)`. Plate-gathers `beta_0[out_idx]`, `beta_1[out_idx]`, `phi[out_idx]` broadcast each output's coefficients across its share of the flat `Resp` plate.

The program returns `beta_1`, an `Out`-indexed plate of real scalars, so the declared codomain is `Val : Real 1`: the per-row value space of the returned coefficients. `Resp` names the plate extent of the response and appears in the signature only as the domain.

## Try it

> The short fits below demonstrate the API. Assess convergence with multiple chains and diagnostics before interpreting a posterior.


### Generating synthetic data

```python
import torch
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/beta_regression.qvr")
model = prog.morphism

N, D = 21, 3
ND = N * D
out_idx = torch.arange(D).repeat(N)
x = torch.randn(ND)

true_beta_0 = torch.tensor([0.5, -0.3, 0.0])
true_beta_1 = torch.tensor([1.0, -1.5, 0.8])
true_phi = torch.tensor([10.0, 20.0, 15.0])
mu_true = torch.sigmoid(true_beta_0[out_idx] + true_beta_1[out_idx] * x)
alpha_true = mu_true * true_phi[out_idx]
beta_true = (1.0 - mu_true) * true_phi[out_idx]
y = torch.distributions.Beta(alpha_true, beta_true).sample()

observations = {"x": x, "y": y, "out_idx": out_idx}
x_in = torch.zeros(ND, 1)
```

### SVI fit

```python
from quivers.inference import AutoNormalGuide, ELBO, SVI

oracle_nll = float(
    -torch.distributions.Beta(alpha_true, beta_true).log_prob(y).mean()
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

N_mcmc, D_mcmc = 11, 3
ND_mcmc = N_mcmc * D_mcmc
out_idx_mcmc = torch.arange(D_mcmc).repeat(N_mcmc)
x_mcmc = x[:ND_mcmc]
y_mcmc = y[:ND_mcmc]
obs_mcmc = {"x": x_mcmc, "y": y_mcmc, "out_idx": out_idx_mcmc}
x_in_mcmc = torch.zeros(ND_mcmc, 1)

torch.manual_seed(2)
kernel = NUTSKernel(step_size=0.05, max_tree_depth=3, target_accept=0.8)
mc = MCMC(kernel, num_warmup=20, num_samples=20, num_chains=1)
result = mc.run(model, x_in_mcmc, obs_mcmc)

print(f"acceptance:  {float(result.acceptance_rates.mean()):.2f}")
print(f"divergences: {int(result.divergence_counts.sum())}")
```


## Categorical perspective

The model is a [Kleisli morphism](https://ncatlab.org/nlab/show/Kleisli+category) in the [Giry monad](https://doi.org/10.1007/BFb0092872)'s Kleisli category whose codomain factors through the [unit interval](https://en.wikipedia.org/wiki/Unit_interval); the per-cell Beta likelihood factors through the link `sigmoid` as a `1 -> G((0, 1))` kernel. The plate-gather `beta_1[out_idx]` is the Kleisli pullback of the `Out`-indexed plate along the fibration `Resp -> Out` carried by the runtime index. [@giry1982categorical]


## Model reference

- Silvia Ferrari and Francisco Cribari-Neto. 2004. Beta regression for modeling rates and proportions. *Journal of Applied Statistics*, 31(7):799–815.
