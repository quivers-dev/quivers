# Gamma Regression

## Overview

A Bayesian [Gamma](https://en.wikipedia.org/wiki/Gamma_distribution)
[generalized linear model](https://en.wikipedia.org/wiki/Generalized_linear_model)
[@nelder1972glm] for strictly positive responses such as durations, costs, or
concentrations. Each of K = 3 categories has its own intercept, slope, and
shape. The log link sets the mean `mu = exp(eta)`, and the shape `nu` fixes the
coefficient of variation at `1 / sqrt(nu)`, so the variance `mu**2 / nu` grows
with the square of the mean. A constant-variance Normal regression does not have this mean-variance relationship. A log-Normal regression with fixed log-scale also has constant coefficient of variation, so that property alone does not distinguish it from a Gamma GLM.

## QVR source

```qvr
# Gamma Regression
#
# A Gamma generalized linear model with a log link and one
# intercept, slope, and shape per category. The linear predictor
# sets the mean mu = exp(eta); the shape nu sets the coefficient
# of variation 1 / sqrt(nu), so the variance is mu**2 / nu.
#
# Generative structure:
#
#   beta_0_k ~ Normal(0, 2)                       per-category intercept
#   beta_1_k ~ Normal(0, 2)                       per-category slope
#   shape_k  ~ Gamma(2, 0.5)                      per-category shape
#   mu_n     = exp(beta_0_{k(n)} + beta_1_{k(n)} * x_n)
#   y_n      ~ Gamma(concentration=shape_{k(n)},
#                    rate=shape_{k(n)} / mu_n)
#
# Gamma(concentration, rate) has mean concentration / rate, so
# rate = shape / mu gives mean mu and variance mu**2 / shape.
#
# Reference: [Nelder and Wedderburn 1972](https://doi.org/10.2307/2344614).

object Cat : FinSet 3
object Resp : FinSet 300
object Val : Real 1

program gamma_regression : Resp -> Val
    sample beta_0 : Cat <- Normal(loc=0.0, scale=2.0)
    sample beta_1 : Cat <- Normal(loc=0.0, scale=2.0)
    sample shape : Cat <- Gamma(concentration=2.0, rate=0.5)

    let b0 = beta_0[cat_idx]
    let b1 = beta_1[cat_idx]
    let nu = shape[cat_idx]
    let mu = exp(b0 + b1 * x)
    let rate = nu / mu

    observe y : Resp <- Gamma(concentration=nu, rate=rate)
    return beta_1

export gamma_regression
```

## Understanding the model

A Gamma observation is a strictly positive amount. Under the
`Gamma(concentration, rate)` convention its mean is `concentration / rate` and
its variance is `concentration / rate**2`. Setting `concentration = nu` and
`rate = nu / mu` thus gives mean `mu` and variance `mu**2 / nu`: the
regression controls the mean through `eta`, while the shape controls how
dispersed the responses are around it, independently of the mean.

Exponentiating the [linear predictor](../tutorials/qvr/02-glms.md#poisson-regression)
keeps `mu` positive, and `beta_1` reads as the change in log mean per unit of
`x`. The category lookup selects which coefficients and shape govern each
flattened response row.

## Walkthrough

Per-category plates `beta_0 : Cat`, `beta_1 : Cat`, and `shape : Cat` carry
one intercept, slope, and shape per category, and the gathers `beta_0[cat_idx]`,
`beta_1[cat_idx]`, and `shape[cat_idx]` broadcast them to the 300 response rows
(100 per category). The let-step `rate = nu / mu` converts the mean into the
family's rate parameter. The Gamma(2, 0.5) prior on the shape has mean 4 and a
long right tail, so it admits both very noisy responses (`nu` near 1, an
exponential-like spread) and tightly concentrated ones.

`return beta_1` exposes one real slope per category. `Cat` indexes those slopes, and `Resp` indexes observations.

## Try it

The blocks below simulate from the model, check the compiled log joint against
a hand-written density, and then recover the parameters with [SVI](../guides/inference-svi.md) (stochastic variational inference) and [NUTS](../guides/inference-mcmc.md) (No-U-Turn Sampler). The input `x_in` has one row, representing one [run over the full data set](../guides/inference-foundations.md#input-rows-are-independent-runs).

### Generating synthetic data

```python
import torch
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/gamma_regression.qvr")
model = prog.morphism

N, K = 100, 3
NK = N * K
cat_idx = torch.arange(K).repeat(N)
x = torch.randn(NK)

true_beta_0 = torch.tensor([1.0, 0.5, -0.5])
true_beta_1 = torch.tensor([0.8, -1.0, 0.3])
true_shape = torch.tensor([2.0, 5.0, 10.0])

mu_true = torch.exp(true_beta_0[cat_idx] + true_beta_1[cat_idx] * x)
nu_true = true_shape[cat_idx]
likelihood_true = torch.distributions.Gamma(nu_true, nu_true / mu_true)
y = likelihood_true.sample()

observations = {"x": x, "y": y, "cat_idx": cat_idx}
x_in = torch.zeros(1, 1)
```

### Checking the density

Replicates drawn at the true parameters should have mean `mu` and variance
`mu**2 / nu`, and the compiled log joint at the true latents must equal the
same density written directly in torch.

```python
replicates = likelihood_true.sample((4000,))
mean_rel_err = ((replicates.mean(0) - mu_true).abs() / mu_true).max()
var_ratio = (replicates.var(0) / (mu_true**2 / nu_true)).mean()
print(f"max relative error of the sample mean: {float(mean_rel_err):.3f}")
print(f"mean ratio of sample to Gamma variance: {float(var_ratio):.3f}")
assert mean_rel_err < 0.1
assert abs(float(var_ratio) - 1.0) < 0.05

Normal = torch.distributions.Normal
Gamma = torch.distributions.Gamma
truth = {"beta_0": true_beta_0, "beta_1": true_beta_1, "shape": true_shape}
oracle = (
    Normal(0.0, 2.0).log_prob(true_beta_0).sum()
    + Normal(0.0, 2.0).log_prob(true_beta_1).sum()
    + Gamma(2.0, 0.5).log_prob(true_shape).sum()
    + likelihood_true.log_prob(y).sum()
)
compiled = model.log_joint(x_in, {**truth, **observations}).sum()
print(f"compiled log joint: {float(compiled):.4f}")
print(f"torch log joint:    {float(oracle):.4f}")
assert torch.allclose(compiled, oracle, rtol=1e-5)
```

### SVI fit

A [mean-field](../guides/inference-svi.md) [`AutoNormalGuide`](../api/inference/guide.md) is fit by
[`SVI`](../api/inference/svi.md). With 100 rows per category, the posterior
standard deviation of a coefficient is at most about `1 / sqrt(nu * N)`, roughly
0.07 for the noisiest category, so the check asks every coefficient's posterior
mean to sit within 0.3 of the truth. The shape is checked on the log scale to
within log(1.5), about three posterior standard deviations of `log nu` at this
sample size.

```python
from quivers.inference import AutoNormalGuide, ELBO, SVI

torch.manual_seed(1)
guide = AutoNormalGuide(model, observed_names={"x", "y", "cat_idx"})
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
assert (post["shape"].log() - true_shape.log()).abs().max() < 0.41
```

### NUTS posterior

[`NUTSKernel`](../api/inference/mcmc.md) samples the same posterior over all
300 rows. Its posterior means should meet the same 0.3 tolerance against the
truth and agree with the SVI posterior means to within 0.15, since the two
methods target one posterior.

```python
from quivers.inference import MCMC, NUTSKernel

torch.manual_seed(2)
kernel = NUTSKernel(step_size=0.05, max_tree_depth=4, target_accept=0.8)
mc = MCMC(kernel, num_warmup=100, num_samples=100, num_chains=1)
result = mc.run(model, x_in, observations)

for k in ("beta_0", "beta_1"):
    s = result.samples[k].reshape(-1, K)
    mean, sd = s.mean(0), s.std(0)
    print(f"{k}: mean {mean.numpy().round(2)}, sd {sd.numpy().round(2)}")
    assert (mean - truth[k]).abs().max() < 0.3
    assert (mean - post[k]).abs().max() < 0.15
print(f"divergences: {int(result.divergence_counts.sum())}")
```

## Categorical perspective

A [probability kernel](https://en.wikipedia.org/wiki/Markov_kernel) assigns a probability distribution to each input. Such kernels compose by integration in the [Kleisli category](https://ncatlab.org/nlab/show/Kleisli+category) of the [Giry monad](https://ncatlab.org/nlab/show/Giry+monad) [@giry1982categorical]. Given the category parameters, each response uses the deterministic link `mu = exp(eta)` followed by `Gamma(nu, nu / mu)`. The gathers `beta_0[cat_idx]` and `shape[cat_idx]` reindex the category parameters along `Resp -> Cat`: each response reads the parameters of its category. Conditional independence gives the product of these response kernels over `Resp`.

Observing data weights this sampling distribution by a likelihood. The resulting measure need not have total mass one, so the scored program uses weighted-measure semantics [@staton2017commutative].
