# Linear Regression with Logistic Noise

## Overview

A linear regression whose additive observation noise follows the [Logistic distribution](https://en.wikipedia.org/wiki/Logistic_distribution) rather than the conventional Normal. The Logistic distribution is symmetric and unimodal with exponential tails, which are heavier than Gaussian tails. [@balakrishnan1992logistic]

Generative structure:

$$
\begin{aligned}
s &\sim \mathrm{HalfNormal}(2),\\
\beta_0 &\sim \mathrm{Normal}(0, 5),\\
\beta_1 &\sim \mathrm{Normal}(0, 2),\\
y_n \mid \beta_0, \beta_1, s &\sim \mathrm{Logistic}(\beta_0 + \beta_1 x_n,\, s).
\end{aligned}
$$

The Logistic density is $f(y; \mu, s) = \frac{e^{-(y - \mu)/s}}{s \bigl(1 + e^{-(y - \mu)/s}\bigr)^2}$ with mean $\mu$ and variance $s^2 \pi^2 / 3$.

## QVR source

```qvr
# Linear Regression with Logistic Noise
#
# A linear regression whose additive observation noise follows the
# Logistic distribution rather than the conventional Normal. The
# Logistic distribution is symmetric and unimodal with
# exponential tails, which are heavier than Gaussian tails. It
# is thus a heavier-tailed alternative when the response has
# more tail mass than a Normal error model allows.
#
# Generative structure:
#
#   scale   ~ HalfNormal(2)                 Logistic scale, half-Normal prior
#   beta_0  ~ Normal(0, 5)                  intercept
#   beta_1  ~ Normal(0, 2)                  slope
#   y_n     ~ Logistic(beta_0 + beta_1 * x_n, scale)
#
# Reference: [Balakrishnan 1992](https://doi.org/10.1201/9781482277098).

object Resp : FinSet 64
object Val : Real 1

program logistic_noise_regression : Resp -> Val
    sample scale <- HalfNormal(scale=2.0)
    sample beta_0 <- Normal(loc=0.0, scale=5.0)
    sample beta_1 <- Normal(loc=0.0, scale=2.0)

    let mu = beta_0 + beta_1 * x

    observe y : Resp <- Logistic(loc=mu, scale=scale)
    return scale

export logistic_noise_regression
```

## Understanding the model

Despite its name, this is a regression for a continuous response. [Binary logistic regression](../tutorials/qvr/02-glms.md#logistic-regression) uses a sigmoid-linked Bernoulli probability. This model
instead adds a continuous Logistic-distributed error to a line. Its scale
`s` is not its standard deviation: the latter is `s * pi / sqrt(3)`.

The inverse-CDF data generator turns a uniform probability into a Logistic
error. Posterior coefficient draws describe uncertainty about the mean line,
while predictive draws add those errors. To compare it with Normal regression,
inspect the distribution of residuals or [replicated extremes](../tutorials/python/08-analysis-pipelines.md#posterior-predictive-checks) as well as the
mean fit. Changing a noise family primarily changes the tail assumptions.

## Walkthrough

The two regression coefficients have Normal priors, and the noise scale has a half-Normal prior with scale 2. The `let mu = ...` expression computes the linear predictor. `observe` evaluates the Logistic density at each response with that location and the shared scale. Backend coverage is recorded in the [support matrix](../transpile-support.md).

`return scale` exposes the positive noise scale. `Resp` indexes the 64 observed responses.

## Try it

The blocks below simulate from the model, check the compiled log joint against
a hand-written density, and then recover the parameters with [SVI](../guides/inference-svi.md) (stochastic variational inference) and [NUTS](../guides/inference-mcmc.md) (No-U-Turn Sampler). The input `x_in` has one row, representing one [run over the full data set](../guides/inference-foundations.md#input-rows-are-independent-runs).

### Generating synthetic data

Draw a predictor, form the linear predictor at known coefficients, and add Logistic noise through the inverse CDF $\mu + s\,\log\bigl(u / (1 - u)\bigr)$ with $u \sim \mathrm{Uniform}(0, 1)$. The `Resp` plate is sized at 64, so the snippet generates 64 rows.

```python
import torch
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/logistic_noise_regression.qvr")
model = prog.morphism

N = 64
true_scale = torch.tensor(0.4)
true_beta_0 = torch.tensor(1.0)
true_beta_1 = torch.tensor(-1.5)

x = torch.randn(N)
mu_true = true_beta_0 + true_beta_1 * x
u = torch.rand(N)
y = mu_true + true_scale * (torch.log(u) - torch.log1p(-u))
observations = {"x": x, "y": y}
x_in = torch.zeros(1, 1)
```

### Checking the log joint

```python
D = torch.distributions
truth = {"scale": true_scale, "beta_0": true_beta_0, "beta_1": true_beta_1}
z = (y - mu_true) / true_scale
logistic_log_density = -z - 2.0 * torch.nn.functional.softplus(-z) - torch.log(true_scale)
oracle = (
    D.HalfNormal(2.0).log_prob(true_scale)
    + D.Normal(0.0, 5.0).log_prob(true_beta_0)
    + D.Normal(0.0, 2.0).log_prob(true_beta_1)
    + logistic_log_density.sum()
)
compiled = model.log_joint(x_in, {**truth, **observations}).sum()
print(f"compiled log joint: {float(compiled):.4f}")
print(f"torch log joint:    {float(oracle):.4f}")
assert torch.allclose(compiled, oracle, rtol=1e-5)
```

### SVI fit

A [mean-field](../guides/inference-svi.md) [`AutoNormalGuide`](../api/inference/guide.md) is fit by
[`SVI`](../api/inference/svi.md). With 64 rows and Logistic scale 0.4 (a residual standard deviation near
0.73), the coefficients' standard errors are near 0.09 and the scale's near
0.04, so the tolerances are about four standard errors: 0.35 for each
coefficient and 0.17 for `scale`.

```python
from quivers.inference import AutoNormalGuide, ELBO, SVI

tolerance = {"beta_0": 0.35, "beta_1": 0.35, "scale": 0.17}

torch.manual_seed(1)
guide = AutoNormalGuide(model, observed_names={"x", "y"})
optim = torch.optim.Adam(guide.parameters(), lr=5e-2)
svi = SVI(model, guide, optim, ELBO(num_particles=1))
for _ in range(1500):
    svi.step(x_in, observations)

with torch.no_grad():
    draws = [guide.rsample(x_in) for _ in range(500)]
post = {k: float(torch.stack([d[k] for d in draws]).mean()) for k in tolerance}
for k, tol in tolerance.items():
    print(f"{k:>8}: posterior mean {post[k]:.3f}, truth {float(truth[k]):.3f}")
    assert abs(post[k] - float(truth[k])) < tol
```

### NUTS posterior

[`NUTSKernel`](../api/inference/mcmc.md) samples the same posterior over all
64 rows. Its posterior means should meet the same tolerances and agree with
the SVI means to within half of each tolerance.

```python
from quivers.inference import MCMC, NUTSKernel

torch.manual_seed(2)
kernel = NUTSKernel(step_size=0.05, max_tree_depth=4, target_accept=0.8)
mc = MCMC(kernel, num_warmup=100, num_samples=100, num_chains=1)
result = mc.run(model, x_in, observations)

for k, tol in tolerance.items():
    mean = float(result.samples[k].mean())
    print(f"{k:>8}: posterior mean {mean:.3f}, truth {float(truth[k]):.3f}")
    assert abs(mean - float(truth[k])) < tol
    assert abs(mean - post[k]) < tol / 2
print(f"divergences: {int(result.divergence_counts.sum())}")
```

## Use cases

Approximation to Normal regression when the analyst suspects modest excess mass at moderate residuals: the Logistic has heavier asymptotic tails than a Normal with the same variance. Common surrogate for the logistic-noise discrete-choice family ($Y_n = \mathbb{1}[\mu_n + \varepsilon_n > 0]$ with $\varepsilon_n$ Logistic recovers binary logistic regression).
