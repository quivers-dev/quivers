# Stochastic Volatility

## Overview

This program factors the one-step conditionals of the log-volatility return model of [@kim1998volatility]. Each latent `h_t` is Normal around an [AR(1)](https://en.wikipedia.org/wiki/Autoregressive_model) mean, and each return is mean-zero Normal with scale `exp(h_t / 2)`. The lagged value `h_prev` is supplied as host data rather than connected to the preceding sampled `h`, so the program conditions on a trajectory of lags; it does not infer one recursively coupled latent chain.

## QVR source

```qvr
# Conditionally Factorized Stochastic Volatility
#
# A one-step-factorized version of the canonical log-volatility
# model. Returns are mean-zero Normal with scale exp(h_t / 2),
# while each h_t is conditioned on a lag h_prev supplied through
# host data. Because the lag is external rather than the previous
# sampled h, this program does not infer one recursively coupled
# latent trajectory.
#
# Generative structure:
#
#   mu       ~ Normal(0, 10)                      log-vol mean
#   phi      ~ Uniform(-1, 1)                     AR(1) coefficient
#   sigma_h  ~ HalfCauchy(2.5)                    log-vol noise
#   h_t      ~ Normal(mu + phi * (h_{t-1} - mu), sigma_h)
#   r_t      ~ Normal(0, exp(h_t / 2))            observed return
#
# Reference: [Kim, Shephard, and Chib 1998](https://doi.org/10.1111/1467-937X.00050).

object Step : FinSet 200
object Val : Real 1

program stochastic_volatility : Step -> Val
    sample mu <- Normal(loc=0.0, scale=10.0)
    sample phi <- Uniform(low=-1.0, high=1.0)
    sample sigma_h <- HalfCauchy(scale=2.5)

    let h_mean = mu + phi * (h_prev - mu)

    sample h : Step <- Normal(loc=h_mean, scale=sigma_h)

    let scale = exp(0.5 * h)

    observe r : Step <- Normal(loc=0.0, scale=scale)
    return phi

export stochastic_volatility
```

## Understanding the model

A volatile period has large fluctuations even when the expected return is
zero. Here `h` represents log variance, so `exp(h / 2)` is the standard
deviation used by the return likelihood. Raising `h` widens the predicted
return distribution without changing its center.

The supplied `h_prev` vector is known to the fit, though it was generated
from a latent process in the simulation. [Conditional](../tutorials/python/02-stochastic-relations.md#observations-and-conditioning)
on those lags, the program fits one-step volatility relations. Inferring a
[coupled volatility path from returns alone](https://mc-stan.org/docs/stan-users-guide/time-series.html#stochastic-volatility-models)
requires treating the earlier states as unknowns too [@stanTimeSeries].
Posterior summaries for this example are conditional on the supplied lag vector.

## Walkthrough

`mu`, `phi`, and `sigma_h` parameterize the AR(1) conditional for log volatility; `phi` is constrained to the [stationarity interval](https://en.wikipedia.org/wiki/Stationary_process) `(-1, 1)`. The identifier `h_prev` is exogenous host data: it is never declared inside the program, so the runtime resolves it from the observations dict at trace time. Conditional on that supplied lag vector, the `h` plate contains independent draws with mean `mu + phi * (h_prev - mu)`. The link `exp(0.5 * h)` then gives the per-step return scale.

The program returns `phi`, a scalar real, so the declared codomain is `Val : Real 1`: the value space of what comes back. `Step` names the plate extent of both the conditionally independent `h` draws and the observed returns, and appears in the signature only as the domain.

## Try it

> The short fits below demonstrate the API. Assess convergence with multiple chains and diagnostics before interpreting a posterior.

### Generating synthetic data

Pick ground-truth log-volatility dynamics, simulate an AR(1) chain `h_t`, then draw mean-zero Normal returns whose scale is `exp(h_t / 2)`. The fit conditions on the simulated lag vector `h_prev`; it does not reconstruct those lags recursively.

```python
import torch
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/stochastic_volatility.qvr")
model = prog.morphism

T = 200
mu_true, phi_true, sigma_h_true = 0.0, 0.95, 0.25
h = torch.zeros(T)
h[0] = mu_true
for t in range(1, T):
    h[t] = mu_true + phi_true * (h[t - 1] - mu_true) + sigma_h_true * torch.randn(())
h_prev = torch.cat([torch.zeros(1), h[:-1]])
returns = torch.exp(0.5 * h) * torch.randn(T)
x_in = torch.zeros(1, 1)
observations = {"r": returns, "h_prev": h_prev}
```

### SVI fit

Reinitialize the program and fit the AR(1) conditional parameters by maximizing the ELBO. The reported oracle evaluates only the return likelihood at the simulated `h`; the ELBO also includes latent-state, prior, and guide terms, so these two values score different quantities.

```python
import torch.distributions as D
from quivers.inference import AutoNormalGuide, ELBO, SVI

torch.manual_seed(0)
prog = load("docs/examples/source/stochastic_volatility.qvr")
model = prog.morphism

guide = AutoNormalGuide(model, observed_names={"r"})
optim = torch.optim.Adam(
    list(model.parameters()) + list(guide.parameters()), lr=5e-2,
)
svi = SVI(model, guide, optim, ELBO())

loss0 = svi.step(x_in, observations)
losses = [svi.step(x_in, observations) for _ in range(400)]
loss_final = sum(losses[-20:]) / 20.0
oracle_ll = D.Normal(0.0, torch.exp(0.5 * h)).log_prob(returns).sum().item()
print(f"initial ELBO loss: {loss0:.1f}")
print(f"final ELBO loss:   {loss_final:.1f}")
print(f"oracle -log p(r):  {-oracle_ll:.1f}")
```

### NUTS posterior

The program declares explicit `sample` sites for `mu`, `phi`, `sigma_h`, and the latent log-volatility plate `h`, so [`NUTSKernel`](../api/inference/mcmc.md#quivers.inference.mcmc.NUTSKernel) samples them directly under small warmup and sample budgets.

```python
from quivers.inference import MCMC, NUTSKernel

torch.manual_seed(0)
prog = load("docs/examples/source/stochastic_volatility.qvr")
model = prog.morphism

kernel = NUTSKernel(step_size=0.05, max_tree_depth=3, target_accept=0.8)
mc = MCMC(kernel, num_warmup=15, num_samples=15, num_chains=1)
result = mc.run(model, x_in, observations)

print("acceptance:", float(result.acceptance_rates.mean()))
print("divergences:", int(result.divergence_counts.sum()))
print(f"mu posterior mean:      {result.samples['mu'].mean().item():.3f}")
print(f"phi posterior mean:     {result.samples['phi'].mean().item():.3f}")
print(f"sigma_h posterior mean: {result.samples['sigma_h'].mean().item():.3f}")
```

## Categorical perspective

The model is a Kleisli morphism over the latent log-volatility plate, composed with a per-step Normal observation kernel whose scale depends on the latent. In the [Giry monad](https://doi.org/10.1007/BFb0092872)'s Kleisli category, `h_prev -> h -> r` is associative Kleisli composition at each row; the SVI guide approximates $p(\mu, \phi, \sigma_h, h \mid r, h_{\mathrm{prev}})$. [@giry1982categorical]

## Model reference

- Sangjoon Kim, Neil Shephard, and Siddhartha Chib. 1998. Stochastic volatility: Likelihood inference and comparison with ARCH models. *The Review of Economic Studies*, 65(3):361–393.
