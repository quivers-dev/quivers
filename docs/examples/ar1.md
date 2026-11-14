# AR(1) Time Series

## Overview

The first-order
[autoregressive](https://en.wikipedia.org/wiki/Autoregressive_model) model:
each observation is a Normal draw centered on a linear function of the
previous observation, with three free parameters (intercept, autoregressive
coefficient, and innovation scale). The program scores the series under its
conditional likelihood: the value that starts the series is held fixed and
every later value is scored under the one-step transition from its
predecessor. It is the reference point for the state-space models in the
gallery and shows the minimum QVR needed to fit a linear time series.

## QVR source

```qvr
# First-Order Autoregressive Model
#
# A first-order autoregressive series with Normal innovations,
# scored under its conditional likelihood: given the value y_0 that
# starts the series, each of the T observed values y_1, ..., y_T is
# Normal around a linear function of its predecessor. This is the
# reference point for the state-space models in the gallery.
#
# Generative structure:
#
#   alpha  ~ Normal(0, 5)                         intercept
#   phi    ~ Uniform(-1, 1)                       stationary region
#   sigma  ~ HalfCauchy(1)                        innovation scale
#   y_t    ~ Normal(alpha + phi * y_{t-1}, sigma) for t = 1, ..., T
#
# Step indexes the T scored transitions. y_prev is the series
# shifted by one, provided as data: row t of y_prev holds y_{t-1}, so
# the first row conditions on y_0, which the likelihood does not
# score. Inside the Uniform(-1, 1) support the series has a
# stationary distribution with mean alpha / (1 - phi) and variance
# sigma^2 / (1 - phi^2).

object Step : FinSet 64
object Val : Real 1

program ar1 : Step -> Val
    sample alpha <- Normal(loc=0.0, scale=5.0)
    sample phi <- Uniform(low=-1.0, high=1.0)
    sample sigma <- HalfCauchy(scale=1.0)

    let mu = alpha + phi * y_prev

    observe y : Step <- Normal(loc=mu, scale=sigma)
    return phi

export ar1
```

## Understanding the model

If `phi = 0.8`, a one-unit deviation in the previous observation shifts the
next conditional mean by `0.8` units. Repeated effects decay when
`abs(phi) < 1`, and the series then has a stationary distribution with mean
`alpha / (1 - phi)` and standard deviation `sigma / sqrt(1 - phi^2)`. The
`Uniform(-1, 1)` prior on `phi` places all of its mass on that region.
`sigma` controls the new variation added at each step.

For a series $y_0, y_1, \ldots, y_T$ the program scores

$$
p(y_1, \ldots, y_T \mid y_0, \alpha, \phi, \sigma)
  = \prod_{t = 1}^{T} \mathcal{N}(y_t \mid \alpha + \phi\, y_{t-1}, \sigma),
$$

the [conditional likelihood](https://mc-stan.org/docs/stan-users-guide/time-series.html#ar1-models)
given the first value [@stanTimeSeries]. It differs from the full likelihood
only by the factor $p(y_0 \mid \alpha, \phi, \sigma)$, whose influence on
the posterior shrinks as the series grows.
[Posterior draws](../tutorials/qvr/01-first-model.md#inspect-the-posterior)
of `phi` describe uncertainty in persistence. To forecast several future
steps, feed each generated observation back as the next step's lag.

## Walkthrough

`alpha` is the intercept of the recurrence, `phi` is the autoregressive coefficient with a `Uniform(-1, 1)` prior over the [stationary region](https://en.wikipedia.org/wiki/Stationary_process), and `sigma` is the innovation scale with a [half-Cauchy](https://en.wikipedia.org/wiki/Cauchy_distribution#Related_distributions) prior. The identifier `y_prev` is never declared inside the program, so the runtime resolves it from the observations mapping at trace time: it is the series shifted by one position. The mean `mu = alpha + phi * y_prev` is the per-step recurrence and the observed `y` is Normal noise around it.

`Step` counts the `T = 64` scored transitions, so `y` and `y_prev` both have 64 rows, and the series they come from has 65 values. Row `t` pairs $y_{t+1}$ with its predecessor $y_t$, and the first value $y_0$ appears only as a lag.

`return phi` exposes the autoregressive coefficient as a real scalar. `Step` indexes the 64 observed transitions.

## Try it

### Generating synthetic data

Pick ground-truth dynamics, draw the starting value from the stationary distribution, and forward-sample `T = 64` transitions. The observed rows and their lags have exactly the `Step` extent the program declares.

```python
import math

import torch
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/ar1.qvr")
model = prog.morphism

T = 64
true_alpha, true_phi, true_sigma = 0.5, 0.7, 0.3
series = torch.zeros(T + 1)
series[0] = true_alpha / (1.0 - true_phi) + true_sigma / math.sqrt(
    1.0 - true_phi**2
) * torch.randn(())
for t in range(1, T + 1):
    series[t] = true_alpha + true_phi * series[t - 1] + true_sigma * torch.randn(())

y = series[1:]
y_prev = series[:-1]
x_in = torch.zeros(1, 1)
observations = {"y": y, "y_prev": y_prev}
```

### Checking the joint density

[`log_joint`](../api/continuous/programs.md) scores the program at given values for every site. At the true parameters it must equal the conditional AR(1) likelihood written directly in PyTorch, plus the three prior terms.

```python
import torch.distributions as D

def ar1_log_joint(alpha, phi, sigma, rows, lags):
    priors = (
        D.Normal(0.0, 5.0).log_prob(torch.tensor(alpha))
        + D.Uniform(-1.0, 1.0).log_prob(torch.tensor(phi))
        + D.HalfCauchy(1.0).log_prob(torch.tensor(sigma))
    )
    return priors + D.Normal(alpha + phi * lags, sigma).log_prob(rows).sum()

truth = {
    "alpha": torch.tensor([true_alpha]),
    "phi": torch.tensor([true_phi]),
    "sigma": torch.tensor([true_sigma]),
    **observations,
}
model_lj = model.log_joint(x_in, truth)
oracle_lj = ar1_log_joint(true_alpha, true_phi, true_sigma, y, y_prev)
print(f"model log joint:  {model_lj.item():.4f}")
print(f"oracle log joint: {oracle_lj.item():.4f}")
assert torch.allclose(model_lj, oracle_lj.reshape(1), atol=1e-3)
```

### SVI fit

Fit `alpha`, `phi`, and `sigma` with a [mean-field](../guides/inference-svi.md) [`AutoNormalGuide`](../api/inference/guide.md#quivers.inference.guides.AutoNormalGuide) and check the posterior means against the truth. The tolerances are about three asymptotic standard errors at `T = 64`: `sqrt((1 - phi^2) / T)` is about `0.09` for `phi` and `sigma / sqrt(2 T)` is about `0.03` for `sigma`, while the intercept is less precisely determined because it trades off against `phi` through the stationary mean.

```python
from quivers.inference import AutoNormalGuide, ELBO, SVI

torch.manual_seed(0)
guide = AutoNormalGuide(model, observed_names={"y"})
optim = torch.optim.Adam(guide.parameters(), lr=2e-2)
svi = SVI(model, guide, optim, ELBO())
for _ in range(1000):
    svi.step(x_in, observations)

with torch.no_grad():
    draws = [guide.rsample(x_in) for _ in range(500)]
svi_means = {
    name: torch.stack([d[name].reshape(-1)[0] for d in draws]).mean().item()
    for name in ("alpha", "phi", "sigma")
}
print({name: round(value, 3) for name, value in svi_means.items()})
assert abs(svi_means["alpha"] - true_alpha) < 0.5
assert abs(svi_means["phi"] - true_phi) < 0.25
assert abs(svi_means["sigma"] - true_sigma) < 0.08
```

### NUTS posterior

The three parameters are the only latent sites, so [`NUTSKernel`](../api/inference/mcmc.md#quivers.inference.mcmc.NUTSKernel) samples them directly. The same tolerances apply to the posterior means. A longer, multi-chain run is needed for well-calibrated intervals.

```python
from quivers.inference import MCMC, NUTSKernel

kernel = NUTSKernel(step_size=0.05, max_tree_depth=4, target_accept=0.8)
mc = MCMC(kernel, num_warmup=100, num_samples=100, num_chains=1, seed=0)
result = mc.run(model, x_in, observations)

nuts_means = {
    name: result.samples[name].mean().item() for name in ("alpha", "phi", "sigma")
}
print("acceptance:", round(float(result.acceptance_rates.mean()), 3))
print("divergences:", int(result.divergence_counts.sum()))
print({name: round(value, 3) for name, value in nuts_means.items()})
assert abs(nuts_means["alpha"] - true_alpha) < 0.5
assert abs(nuts_means["phi"] - true_phi) < 0.25
assert abs(nuts_means["sigma"] - true_sigma) < 0.08
```

## Categorical perspective

The conditional AR(1) joint is $p(\alpha, \phi, \sigma) \prod_{t \ge 1} \mathcal{N}(y_t \mid \alpha + \phi y_{t-1}, \sigma)$, where $\sigma$ is the standard deviation. A [probability kernel](https://en.wikipedia.org/wiki/Markov_kernel) assigns a probability distribution to each input. Such kernels compose by integration in the [Kleisli category](https://ncatlab.org/nlab/show/Kleisli+category) of the [Giry monad](https://ncatlab.org/nlab/show/Giry+monad) [@giry1982categorical]. The `Step` plate multiplies the Normal likelihood factors evaluated at the observed lags. This factorization does not make successive observations independent in the generative chain, where each newly drawn observation becomes the next lag.

Observing data weights this sampling distribution by a likelihood. The resulting measure need not have total mass one, so the scored program uses weighted-measure semantics [@staton2017commutative].
