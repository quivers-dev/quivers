# AR(1) Time Series

## Overview

The stationary first-order
[autoregressive](https://en.wikipedia.org/wiki/Autoregressive_model) model:
each observation is a Normal draw centered on a linear function of the
previous observation, with three free parameters (intercept, autoregressive
coefficient, and innovation scale). The program scores the series under its
exact likelihood, so the first value comes from the
[stationary distribution](https://en.wikipedia.org/wiki/Stationary_distribution)
and every later value from the one-step transition. The lag is read from the
observed series inside the program rather than supplied as a second input.
This is the reference point for the latent-state
[stochastic-volatility](stochastic-volatility.md) example, which uses the same
lag idiom on an unobserved series.

## QVR source

```qvr
# First-Order Autoregressive Model
#
# A stationary first-order autoregressive series with Normal
# innovations, scored under its exact likelihood: the first value
# is drawn from the stationary distribution and every later value
# from the one-step transition given its predecessor. This is the
# reference point for the latent-state time-series models in the
# gallery.
#
# Generative structure:
#
#   alpha  ~ Normal(0, 5)                         intercept
#   phi    ~ Uniform(-1, 1)                       stationary region
#   sigma  ~ HalfCauchy(1)                        innovation scale
#   y_0    ~ Normal(alpha / (1 - phi), sigma / sqrt(1 - phi^2))
#   y_t    ~ Normal(alpha + phi * y_{t-1}, sigma) for t >= 1
#
# The observe step scores every y_t under the stationary marginal.
# The transition score then replaces that factor at each t >= 1 with
# the transition density: it adds log N(y_t | alpha + phi * y_{t-1},
# sigma) - log N(y_t | stationary), so the total is the exact joint
# log density. The lag y_{t-1} is read from the observed series
# itself through a shifted factor expression.

object Step : FinSet 64
object Val : Real 1

program ar1 : Step -> Val
    sample alpha <- Normal(loc=0.0, scale=5.0)
    sample phi <- Uniform(low=-1.0, high=1.0)
    sample sigma <- HalfCauchy(scale=1.0)

    let stationary_mean = alpha / (1.0 - phi)
    let stationary_scale = sigma / sqrt(1.0 - phi * phi)

    observe y : Step <- Normal(loc=stationary_mean, scale=stationary_scale)

    let y_lag = factor t : Step in y[max(t - 1, 0)]
    let is_transition = factor t : Step in real(min(t, 1))
    let z_transition = (y - alpha - phi * y_lag) / sigma
    let z_stationary = (y - stationary_mean) / stationary_scale
    score transition = sum(is_transition * (0.5 * z_stationary * z_stationary - 0.5 * z_transition * z_transition + log(stationary_scale) - log(sigma)))
    return phi

export ar1
```

## Understanding the model

If `phi = 0.8`, a one-unit deviation in the previous observation shifts the
next conditional mean by `0.8` units. Repeated effects decay when
`abs(phi) < 1`, and the series then settles into a stationary distribution
with mean `alpha / (1 - phi)` and standard deviation
`sigma / sqrt(1 - phi^2)`. `sigma` controls the new variation added at each
step.

A series observed from an arbitrary point in its history is a draw from that
stationary distribution at its first position, so the exact likelihood is

$$
p(y \mid \alpha, \phi, \sigma)
  = \mathcal{N}\!\left(y_0 \,\middle|\, \tfrac{\alpha}{1 - \phi}, \tfrac{\sigma}{\sqrt{1 - \phi^2}}\right)
    \prod_{t \ge 1} \mathcal{N}(y_t \mid \alpha + \phi\, y_{t-1}, \sigma).
$$

Dropping the first factor gives the conditional likelihood, which treats
`y_0` as fixed [@stanTimeSeries]. Keeping it lets the first observation
inform `alpha` and `phi` through the stationary mean and variance, which is
the information the `Uniform(-1, 1)` prior on `phi` makes available: the
stationary distribution exists only inside that interval.
[Posterior draws](../tutorials/qvr/01-first-model.md#inspect-the-posterior)
of `phi` describe uncertainty in persistence.

## Walkthrough

`alpha` is the intercept of the recurrence, `phi` is the autoregressive coefficient with a `Uniform(-1, 1)` prior over the [stationary region](https://en.wikipedia.org/wiki/Stationary_process), and `sigma` is the innovation scale with a [half-Cauchy](https://en.wikipedia.org/wiki/Cauchy_distribution#Related_distributions) prior. The two `let` steps compute the stationary mean and scale.

`observe y : Step <- Normal(...)` scores every position under the stationary marginal. That is the correct factor at `t = 0` and the wrong one at every later position, where the transition density belongs. The [`score`](../reference/qvr/probabilistic-programs.md#step-reference) step swaps them. The [factor expression](../guides/dsl-programs-and-lets.md#factor-expressions-assembling-indexed-tensors) `factor t : Step in y[max(t - 1, 0)]` builds the lagged series from the observed `y` itself, and `factor t : Step in real(min(t, 1))` is the indicator of `t >= 1`. With `z_transition` and `z_stationary` the standardized residuals under the two densities, the Normal log density difference at each position is

$$
\log \mathcal{N}(y_t \mid \text{transition}) - \log \mathcal{N}(y_t \mid \text{stationary})
  = \tfrac{1}{2} z_{\text{stationary}}^2 - \tfrac{1}{2} z_{\text{transition}}^2
    + \log \tfrac{\sigma}{\sqrt{1-\phi^2}} - \log \sigma,
$$

so `score transition` adds exactly the masked sum of these differences, and the program's joint density is the exact likelihood above times the priors.

The program returns `phi`, a scalar real, so the declared codomain is `Val : Real 1`: the value space of what comes back, not the index set the series is plated over. `Step` names that plate extent and appears in the signature only as the domain.

## Try it

### Generating synthetic data

Pick ground-truth dynamics, draw the first value from the stationary distribution, and forward-sample the recurrence for the remaining `T - 1` steps. The series has exactly the `Step` extent the program declares.

```python
import math

import torch
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/ar1.qvr")
model = prog.morphism

T = 64
true_alpha, true_phi, true_sigma = 0.5, 0.7, 0.3
stationary_mean = true_alpha / (1.0 - true_phi)
stationary_scale = true_sigma / math.sqrt(1.0 - true_phi**2)

y = torch.zeros(T)
y[0] = stationary_mean + stationary_scale * torch.randn(())
for t in range(1, T):
    y[t] = true_alpha + true_phi * y[t - 1] + true_sigma * torch.randn(())

x_in = torch.zeros(1, 1)
observations = {"y": y}
```

### Checking the joint density

[`log_joint`](../api/continuous/programs.md) scores the program at given values for every site. At the true parameters it must equal the exact AR(1) likelihood written directly in PyTorch, plus the three prior terms.

```python
import torch.distributions as D

def ar1_log_joint(alpha, phi, sigma, series):
    stationary = D.Normal(alpha / (1.0 - phi), sigma / math.sqrt(1.0 - phi**2))
    transition = D.Normal(alpha + phi * series[:-1], sigma)
    priors = (
        D.Normal(0.0, 5.0).log_prob(torch.tensor(alpha))
        + D.Uniform(-1.0, 1.0).log_prob(torch.tensor(phi))
        + D.HalfCauchy(1.0).log_prob(torch.tensor(sigma))
    )
    return priors + stationary.log_prob(series[0]) + transition.log_prob(series[1:]).sum()

truth = {
    "alpha": torch.tensor([true_alpha]),
    "phi": torch.tensor([true_phi]),
    "sigma": torch.tensor([true_sigma]),
    "y": y,
}
model_lj = model.log_joint(x_in, truth)
oracle_lj = ar1_log_joint(true_alpha, true_phi, true_sigma, y)
print(f"model log joint:  {model_lj.item():.4f}")
print(f"oracle log joint: {oracle_lj.item():.4f}")
assert torch.allclose(model_lj, oracle_lj.reshape(1), atol=1e-3)
```

### SVI fit

Fit `alpha`, `phi`, and `sigma` with a mean-field [`AutoNormalGuide`](../api/inference/guide.md#quivers.inference.guides.AutoNormalGuide) and check the posterior means against the truth. The tolerances are about three asymptotic standard errors at `T = 64`: `sqrt((1 - phi^2) / T)` is about `0.09` for `phi` and `sigma / sqrt(2 T)` is about `0.03` for `sigma`, while the intercept is less precisely determined because it trades off against `phi` through the stationary mean.

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

The three parameters are the only latent sites, so [`NUTSKernel`](../api/inference/mcmc.md#quivers.inference.mcmc.NUTSKernel) samples them directly under the exact likelihood. The same tolerances apply to the posterior means; a longer, multi-chain run is needed for well-calibrated intervals.

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

The model is a Kleisli morphism in the [Giry monad](https://doi.org/10.1007/BFb0092872)'s [Kleisli category](https://ncatlab.org/nlab/show/Kleisli+category) whose denotation is the joint $p(\alpha, \phi, \sigma)\, \mathcal{N}(y_0 \mid \text{stationary}) \prod_{t \ge 1} \mathcal{N}(y_t \mid \alpha + \phi y_{t-1}, \sigma)$. The `Step`-indexed observe is the [right Kan extension](https://ncatlab.org/nlab/show/Kan+extension) of the per-step stationary Normal kernel along the trivial projection $\mathrm{Step} \to \mathbf{1}$, and the `score` step reweights that product by a density ratio, which turns the conditionally independent plate into the Markov chain the recurrence describes. [@giry1982categorical]

## Sources and further reading

- *Time-Series Models*, Stan User's Guide [@stanTimeSeries].
- *A categorical approach to probability theory* [@giry1982categorical].
