# AR(1) Time Series

## Overview

The canonical first-order
[autoregressive](https://en.wikipedia.org/wiki/Autoregressive_model) model:
each observation is a Normal draw centered on a linear function of the previous
observation, with three free parameters (intercept, autoregressive coefficient,
and noise scale). The model is the reference point for the more elaborate
state-space and stochastic-volatility examples in the gallery; it shows the
minimum QVR syntax needed to fit a stationary linear time series.

## QVR source

```qvr
# First-Order Autoregressive Model
#
# A first-order autoregressive time series with Normal noise.
# Conditioning on an observed series indexed by Step gives a
# posterior over (alpha, phi, sigma); this is
# the canonical reference point for the more elaborate
# state-space models in the gallery.
#
# Generative structure:
#
#   alpha  ~ Normal(0, 5)
#   phi    ~ Uniform(-1, 1)                       stationarity constraint
#   sigma  ~ HalfCauchy(1)
#   y_t    ~ Normal(alpha + phi * y_{t-1}, sigma) per-step observation
#
# The Uniform(-1, 1) prior on phi enforces the AR stationarity
# condition. A stationary initialization would have mean
# alpha / (1 - phi) and variance sigma^2 / (1 - phi^2); the
# runnable example instead conditions on an explicit lag vector.

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
next conditional mean by `0.8` units. Repeated effects decay when `abs(phi) < 1`;
the stationary mean is `alpha / (1 - phi)`, not generally `alpha` itself.
`sigma` controls the new variation added at each step.

The fit receives both the current series and its [lagged values](../tutorials/qvr/05-time-series.md#plates-revisited).
The likelihood is conditional on those lags, including the [initial value](https://mc-stan.org/docs/stan-users-guide/time-series.html#ar1-models)
[@stanTimeSeries]. [Posterior draws](../tutorials/qvr/01-first-model.md#inspect-the-posterior)
of `phi` describe uncertainty in persistence. To forecast several future
steps, feed each generated observation back as the next step's lag.

## Walkthrough

`alpha` is the intercept of the AR(1) recurrence, `phi` is the autoregressive coefficient constrained to the [stationarity interval](https://en.wikipedia.org/wiki/Stationary_process) `(-1, 1)` via a `Uniform` prior, and `sigma` is the per-step Normal scale with a [half-Cauchy](https://en.wikipedia.org/wiki/Cauchy_distribution#Related_distributions) prior. The identifier `y_prev` is exogenous host-data: it is never declared inside the program, so the runtime resolves it from the observations dict at trace time, where the caller supplies the lagged response series. The mean `mu = alpha + phi * y_prev` is then the per-step recurrence and the observed series `y` is Normal noise around it.

The program returns `phi`, a scalar real, so the declared codomain is `Val : Real 1`: the value space of what comes back, not the index set the series is plated over. `Step` names that plate extent, and it appears in the signature only as the domain.

## Try it

> The short fits below demonstrate the API. Assess convergence with multiple chains and diagnostics before interpreting a posterior.

### Generating synthetic data

Pick ground-truth dynamics and forward-sample a single AR(1) trajectory of length `T`. The generated series starts at the stationary mean, while the model's lag vector uses a zero boundary at its first entry. Thus the first likelihood row has mean `alpha`, not the stationary mean.

```python
import torch
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/ar1.qvr")
model = prog.morphism

T = 64
alpha_true, phi_true, sigma_true = 0.5, 0.7, 0.3
y = torch.zeros(T)
y[0] = alpha_true / (1.0 - phi_true)
for t in range(1, T):
    y[t] = alpha_true + phi_true * y[t - 1] + sigma_true * torch.randn(())
y_prev = torch.cat([torch.zeros(1), y[:-1]])
x_in = torch.zeros(T, 1)
observations = {"y": y, "y_prev": y_prev}
```

### SVI fit

Re-initialize the model so the optimizer starts from the prior, then fit `alpha`, `phi`, and `sigma` by maximizing the ELBO. Compare the negative ELBO across optimization steps. The oracle likelihood is a separate reference score; prior and guide terms prevent treating it as the loss target.

```python
import torch.distributions as D
from quivers.inference import AutoNormalGuide, ELBO, SVI

torch.manual_seed(0)
prog = load("docs/examples/source/ar1.qvr")
model = prog.morphism

guide = AutoNormalGuide(model, observed_names={"y"})
optim = torch.optim.Adam(
    list(model.parameters()) + list(guide.parameters()), lr=1e-2,
)
svi = SVI(model, guide, optim, ELBO())

loss0 = svi.step(x_in, observations)
losses = [svi.step(x_in, observations) for _ in range(800)]
loss_final = sum(losses[-20:]) / 20.0
oracle_ll = D.Normal(
    alpha_true + phi_true * y_prev, sigma_true,
).log_prob(y).sum().item()
print(f"initial ELBO loss: {loss0:.1f}")
print(f"final ELBO loss:   {loss_final:.1f}")
print(f"oracle -log p(y):  {-oracle_ll:.1f}")
```

### NUTS posterior

The model declares explicit `sample` sites for `alpha`, `phi`, and `sigma`, so [`NUTSKernel`](../api/inference/mcmc.md#quivers.inference.mcmc.NUTSKernel) samples them directly. Small budgets keep the snippet inside tens of seconds for an API demonstration; longer, diagnosed runs are needed for posterior interpretation.

```python
from quivers.inference import MCMC, NUTSKernel

torch.manual_seed(0)
prog = load("docs/examples/source/ar1.qvr")
model = prog.morphism

kernel = NUTSKernel(step_size=0.05, max_tree_depth=3, target_accept=0.8)
mc = MCMC(kernel, num_warmup=15, num_samples=15, num_chains=1)
result = mc.run(model, x_in, observations)

print("acceptance:", float(result.acceptance_rates.mean()))
print("divergences:", int(result.divergence_counts.sum()))
for name, samples in result.samples.items():
    print(f"{name}: posterior mean = {samples.mean().item():.3f}")
```

## Categorical perspective

The model is a Kleisli morphism in the [Giry monad](https://doi.org/10.1007/BFb0092872)'s [Kleisli category](https://ncatlab.org/nlab/show/Kleisli+category) whose denotation is the standard AR(1) joint $p(\alpha, \phi, \sigma) \prod_t \mathcal{N}(y_t \mid \alpha + \phi y_{t-1}, \sigma)$. The `Step`-indexed plate is the [right Kan extension](https://ncatlab.org/nlab/show/Kan+extension) of the per-step Normal kernel along the trivial projection $\mathrm{Step} \to \mathbf{1}$. [@giry1982categorical]

## Sources and further reading

- *A categorical approach to probability theory* [@giry1982categorical].
