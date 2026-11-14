# Multi-Output Beta Regression

## Overview

A multi-output beta regression [@ferrari2004beta] for response variables on the open unit interval. Each output dimension carries its own logit-link coefficients and precision. The per-cell mean is mapped through the [sigmoid](https://en.wikipedia.org/wiki/Logistic_function) link to the unit interval, and the [Beta](https://en.wikipedia.org/wiki/Beta_distribution) likelihood is reparameterized in mean / precision form.

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

object Out : FinSet 3
object Resp : FinSet 300
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
    let beta_p = sigmoid(-eta) * p

    observe y : Resp <- Beta(concentration1=alpha, concentration0=beta_p)
    return beta_1

export beta_regression
```

## Understanding the model

A fraction such as time spent on a task must stay between zero and one.
A [linear predictor](../tutorials/qvr/02-glms.md#logistic-regression) cannot ensure this, so `sigmoid` converts it to a valid
mean. With `mu = 0.5`, increasing `phi` concentrates outcomes near one-half. With the same `phi`, changing `mu` moves their center. Exact zeros and ones
need a different observation model or an explicit boundary treatment.

For multiple outputs, [`out_idx`](../tutorials/qvr/09-indexed-data.md#5-transfer-the-pattern-to-statistical-data) tells each flattened response row which
coefficient and precision to use. If a row belongs to output 2, it gathers
`beta_0[2]`, `beta_1[2]`, and `phi[2]`. Posterior slopes describe changes on
the log-odds scale of the mean, so compare predicted fractions when the
scientific question is about differences on the original response scale.

## Walkthrough

Per-output coefficient plates `beta_0 : Out` and `beta_1 : Out` carry one coefficient per response dimension. Per-output precision `phi : Out` permits heterogeneous dispersion across the response axis. The per-cell linear predictor `eta = b0 + b1 * x` is mapped to the unit interval via `mu = sigmoid(eta)`. The mean / precision form of the Beta is `Beta(mu * phi, (1 - mu) * phi)`, giving mean `mu` and variance `mu * (1 - mu) / (1 + phi)`. Plate-gathers `beta_0[out_idx]`, `beta_1[out_idx]`, `phi[out_idx]` broadcast each output's coefficients across its share of the flat `Resp` [plate](https://en.wikipedia.org/wiki/Plate_notation), 100 rows per output.

The complementary concentration is evaluated as `sigmoid(-eta) * phi`. The identity
`1 - sigmoid(eta) = sigmoid(-eta)` preserves the model, while direct evaluation of
the right side avoids subtracting a rounded value of one at large positive predictors.

`return beta_1` exposes one real slope per output. `Out` indexes those slopes, and `Resp` indexes observations.

## Try it

The blocks below simulate from the model, check the compiled log joint against
a hand-written density, and then recover the parameters with [SVI](../guides/inference-svi.md) (stochastic variational inference) and [NUTS](../guides/inference-mcmc.md) (No-U-Turn Sampler). The input `x_in` has one row, representing one [run over the full data set](../guides/inference-foundations.md#input-rows-are-independent-runs).

### Generating synthetic data

```python
import torch
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/beta_regression.qvr")
model = prog.morphism

N, D = 100, 3
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
x_in = torch.zeros(1, 1)
```

### Checking the log joint

```python
D_ = torch.distributions
truth = {"beta_0": true_beta_0, "beta_1": true_beta_1, "phi": true_phi}
oracle = (
    D_.Normal(0.0, 5.0).log_prob(true_beta_0).sum()
    + D_.Normal(0.0, 5.0).log_prob(true_beta_1).sum()
    + D_.HalfCauchy(2.5).log_prob(true_phi).sum()
    + D_.Beta(alpha_true, beta_true).log_prob(y).sum()
)
compiled = model.log_joint(x_in, {**truth, **observations}).sum()
print(f"compiled log joint: {float(compiled):.4f}")
print(f"torch log joint:    {float(oracle):.4f}")
assert torch.allclose(compiled, oracle, rtol=1e-5)
```

### SVI fit

A [mean-field](../guides/inference-svi.md) [`AutoNormalGuide`](../api/inference/guide.md) is fit by
[`SVI`](../api/inference/svi.md). With 100 rows per output and precisions of
10 to 20, each coefficient's posterior standard deviation is below 0.1, so the
check asks every coefficient's posterior mean to sit within 0.3 of the truth.
The precision's relative standard error is near `sqrt(2 / N)`, about 0.14, so
`phi` is checked on the log scale to within log(1.5).

```python
from quivers.inference import AutoNormalGuide, ELBO, SVI

torch.manual_seed(1)
guide = AutoNormalGuide(model, observed_names={"x", "y", "out_idx"})
optim = torch.optim.Adam(guide.parameters(), lr=5e-2)
svi = SVI(model, guide, optim, ELBO(num_particles=1))
for _ in range(1500):
    svi.step(x_in, observations)

with torch.no_grad():
    draws = [guide.rsample(x_in) for _ in range(500)]
post = {k: torch.stack([d[k] for d in draws]).mean(0) for k in truth}
for k in truth:
    print(f"{k:>6}: posterior mean {post[k].numpy().round(2)}, truth {truth[k].numpy()}")

assert (post["beta_0"] - true_beta_0).abs().max() < 0.3
assert (post["beta_1"] - true_beta_1).abs().max() < 0.3
assert (post["phi"].log() - true_phi.log()).abs().max() < 0.41
```

### NUTS posterior

[`NUTSKernel`](../api/inference/mcmc.md) samples the same posterior over all
300 rows. Its posterior means should meet the same tolerances and agree with
the SVI means to within 0.15 for the coefficients.

```python
from quivers.inference import MCMC, NUTSKernel

torch.manual_seed(2)
kernel = NUTSKernel(step_size=0.05, max_tree_depth=4, target_accept=0.8)
mc = MCMC(kernel, num_warmup=100, num_samples=100, num_chains=1)
result = mc.run(model, x_in, observations)

for k in ("beta_0", "beta_1"):
    mean = result.samples[k].reshape(-1, D).mean(0)
    print(f"{k}: posterior mean {mean.numpy().round(2)}, truth {truth[k].numpy()}")
    assert (mean - truth[k]).abs().max() < 0.3
    assert (mean - post[k]).abs().max() < 0.15
phi_nuts = result.samples["phi"].reshape(-1, D).mean(0)
print(f"phi: posterior mean {phi_nuts.numpy().round(1)}, truth {true_phi.numpy()}")
assert (phi_nuts.log() - true_phi.log()).abs().max() < 0.41
print(f"divergences: {int(result.divergence_counts.sum())}")
```

## Categorical perspective

A [probability kernel](https://en.wikipedia.org/wiki/Markov_kernel) assigns a probability distribution to each input. Such kernels compose by integration in the [Kleisli category](https://ncatlab.org/nlab/show/Kleisli+category) of the [Giry monad](https://ncatlab.org/nlab/show/Giry+monad) [@giry1982categorical]. Each response's Beta [kernel](https://en.wikipedia.org/wiki/Markov_kernel) maps its covariates and parameters to a distribution on the open unit interval $(0,1)$, with mean given by `sigmoid`. The gather `beta_1[out_idx]` is [reindexing](https://ncatlab.org/nlab/show/reindexing): precomposition with the map `Resp -> Out`, assigning each response the coefficient for its output.

Observing data weights this sampling distribution by a likelihood. The resulting measure need not have total mass one, so the scored program uses weighted-measure semantics [@staton2017commutative].
