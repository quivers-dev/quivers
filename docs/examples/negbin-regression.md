# Multi-Output Negative Binomial Regression

## Overview

A multi-output [negative-binomial](https://en.wikipedia.org/wiki/Negative_binomial_distribution)
regression in the NB2 form: each output has its own intercept, slope, and
dispersion, the log link sets the count mean `mu = exp(eta)`, and the
dispersion `phi` sets the variance `mu + mu**2 / phi`. As `phi` grows the
variance approaches `mu` and the model approaches
[Poisson regression](../tutorials/qvr/02-glms.md#poisson-regression). A
small `phi` allows counts far more variable than a Poisson with the same mean. [@cameron1986econometric]

## QVR source

```qvr
# Multi-Output Negative Binomial Regression
#
# An NB2 negative-binomial regression with one intercept, one
# slope, and one dispersion per output. The log link maps the
# linear predictor to the count mean mu = exp(eta); the dispersion
# phi sets the extra-Poisson variance mu + mu**2 / phi.
#
# Generative structure:
#
#   beta_0_d     ~ Normal(0, 5)                   per-output intercept
#   beta_1_d     ~ Normal(0, 5)                   per-output slope
#   phi_d        ~ Gamma(2, 0.5)                  per-output dispersion
#   mu_n         = exp(beta_0_{d(n)} + beta_1_{d(n)} * x_n)
#   y_n          ~ NegativeBinomial(total_count=phi_{d(n)},
#                                   probs=mu_n / (phi_{d(n)} + mu_n))
#
# NegativeBinomial(total_count, probs) has mean
# total_count * probs / (1 - probs), so probs = mu / (phi + mu)
# gives mean mu and variance mu + mu**2 / phi (the NB2 form).

object Out : FinSet 3
object Resp : FinSet 300
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
    let probs = mu / (disp + mu)

    observe y : Resp <- NegativeBinomial(total_count=disp, probs=probs)
    return beta_1

export negbin_regression
```

## Understanding the model

[Poisson variation](../tutorials/qvr/02-glms.md#poisson-regression) ties the
variance of a count to its mean. A negative binomial allows extra variation, as
if an underlying Poisson rate itself varied across observations. The family's
two parameters, a count `total_count` and a probability `probs`, jointly set its
mean, so the regression has to choose `probs` with the family's convention in
view.

`NegativeBinomial(total_count, probs)` counts the successes observed before
`total_count` failures when each trial succeeds with probability `probs`. Its
mean is `total_count * probs / (1 - probs)`. Setting `total_count = phi` and
`probs = mu / (phi + mu)` makes `probs / (1 - probs) = mu / phi`, so the mean is
exactly `mu` and the variance is `mu + mu**2 / phi`. Raising the linear
predictor `eta` thus raises the expected count, and `beta_1` reads as the change
in log mean per unit of `x`, as in any log-link count regression.

## Walkthrough

Per-output coefficient and dispersion plates broadcast to the 300 response rows
through `out_idx` gathers: 100 rows for each of the three outputs. The per-row
linear predictor `eta = b0 + b1 * x` passes through the log link `exp` to give
the mean `mu`, and `probs = mu / (disp + mu)` converts that mean into the
family's probability parameter. The Gamma(2, 0.5) prior on the dispersion has
mean 4 and a long right tail, so it allows anything from strong overdispersion
to counts that are close to Poisson.

`return beta_1` exposes one real slope per output. `Out` indexes those slopes, and `Resp` indexes observations.

## Try it

The blocks below simulate from the model, check the compiled log joint against
a hand-written density, and then recover the coefficients with [SVI](../guides/inference-svi.md) (stochastic variational inference) and [NUTS](../guides/inference-mcmc.md) (No-U-Turn Sampler).
The input `x_in` has one row, representing one [run over the full data set](../guides/inference-foundations.md#input-rows-are-independent-runs).

### Generating synthetic data

```python
import torch
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/negbin_regression.qvr")
model = prog.morphism

N, D = 100, 3
ND = N * D
out_idx = torch.arange(D).repeat(N)
x = torch.randn(ND)

true_beta_0 = torch.tensor([1.0, 0.5, 2.0])
true_beta_1 = torch.tensor([0.5, -0.3, 0.8])
true_dispersion = torch.tensor([5.0, 10.0, 3.0])

mu_true = torch.exp(true_beta_0[out_idx] + true_beta_1[out_idx] * x)
disp_true = true_dispersion[out_idx]
likelihood_true = torch.distributions.NegativeBinomial(
    total_count=disp_true, probs=mu_true / (disp_true + mu_true),
)
y = likelihood_true.sample()

observations = {"x": x, "y": y, "out_idx": out_idx}
x_in = torch.zeros(1, 1)
```

### Checking the density

The NB2 parameterization should give each row mean `mu` and variance
`mu + mu**2 / phi`. Simulating many replicates at the true parameters checks
both moments, and the compiled log joint at the true latents must equal the
same density written directly in torch.

```python
replicates = likelihood_true.sample((4000,))
mean_rel_err = ((replicates.mean(0) - mu_true).abs() / mu_true).max()
var_ratio = (replicates.var(0) / (mu_true + mu_true**2 / disp_true)).mean()
print(f"max relative error of the sample mean: {float(mean_rel_err):.3f}")
print(f"mean ratio of sample to NB2 variance:  {float(var_ratio):.3f}")
assert mean_rel_err < 0.1
assert abs(float(var_ratio) - 1.0) < 0.05

Normal = torch.distributions.Normal
Gamma = torch.distributions.Gamma
truth = {
    "beta_0": true_beta_0,
    "beta_1": true_beta_1,
    "dispersion": true_dispersion,
}
oracle = (
    Normal(0.0, 5.0).log_prob(true_beta_0).sum()
    + Normal(0.0, 5.0).log_prob(true_beta_1).sum()
    + Gamma(2.0, 0.5).log_prob(true_dispersion).sum()
    + likelihood_true.log_prob(y).sum()
)
compiled = model.log_joint(x_in, {**truth, **observations}).sum()
print(f"compiled log joint: {float(compiled):.4f}")
print(f"torch log joint:    {float(oracle):.4f}")
assert torch.allclose(compiled, oracle, rtol=1e-5)
```

### SVI fit

A [mean-field](../guides/inference-svi.md) [`AutoNormalGuide`](../api/inference/guide.md) is fit by
[`SVI`](../api/inference/svi.md). With 100 rows per output, the posterior
standard deviation of each coefficient is roughly 0.08, so the recovery check
asks every coefficient's posterior mean to sit within 0.3 of the truth (about
four standard deviations). The dispersion is weakly identified at this sample
size (its posterior standard deviation is of the same order as its value), so it
is checked only to within a factor of two.

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
    print(f"{k:>10}: posterior mean {post[k].numpy().round(2)}, truth {truth[k].numpy()}")

assert (post["beta_0"] - true_beta_0).abs().max() < 0.3
assert (post["beta_1"] - true_beta_1).abs().max() < 0.3
assert (post["dispersion"].log() - true_dispersion.log()).abs().max() < 0.7
```

### NUTS posterior

[`NUTSKernel`](../api/inference/mcmc.md) samples the same posterior over all
300 rows. Its posterior means should meet the same 0.3 tolerance against the
truth and agree with the SVI posterior means to within 0.15, since the two
methods target one posterior.

The MCMC run uses float64. The broad coefficient priors can make
`mu / (disp + mu)` so close to one that float32 rounds it to the excluded
endpoint during the preliminary recording run. Converting the existing data
preserves every observation and retains these probabilities in this seeded run.
Finite precision still limits how close a probability can be to one.

```python
from quivers.inference import MCMC, NUTSKernel

torch.set_default_dtype(torch.float64)
mc_observations = {
    name: value.double() if value.is_floating_point() else value
    for name, value in observations.items()
}
torch.manual_seed(2)
kernel = NUTSKernel(step_size=0.05, max_tree_depth=4, target_accept=0.8)
mc = MCMC(kernel, num_warmup=100, num_samples=100, num_chains=1)
result = mc.run(model, x_in.double(), mc_observations)

for k in ("beta_0", "beta_1"):
    s = result.samples[k].reshape(-1, D)
    mean, sd = s.mean(0), s.std(0)
    print(f"{k}: mean {mean.numpy().round(2)}, sd {sd.numpy().round(2)}")
    assert (mean - truth[k]).abs().max() < 0.3
    assert (mean - post[k]).abs().max() < 0.15
print(f"divergences: {int(result.divergence_counts.sum())}")
```

## Categorical perspective

The negative binomial is the Gamma-Poisson
[mixture](https://en.wikipedia.org/wiki/Compound_probability_distribution): a
`Poisson(rate)` [kernel](https://en.wikipedia.org/wiki/Markov_kernel) composed after `rate ~ Gamma(concentration=phi,
rate=phi / mu)` marginalizes to `NegativeBinomial(total_count=phi,
probs=mu / (phi + mu))`. The Gamma mixing kernel has mean `mu` and variance
`mu**2 / phi`, which is where the NB2 variance `mu + mu**2 / phi` comes from:
`mu` from the Poisson and `mu**2 / phi` from the varying rate. The program uses
the closed-form marginal per row rather than sampling the per-row rate.
