# Weibull Survival Regression

## Overview

A parametric [survival regression](https://en.wikipedia.org/wiki/Survival_analysis) with a [Weibull](https://en.wikipedia.org/wiki/Weibull_distribution) baseline [@klein2003survival]. Each item carries a covariate `x_i` whose linear contribution scales the Weibull rate; the shape parameter `k` governs whether the hazard is decreasing (k < 1), constant (k = 1, exponential), or increasing (k > 1).

## QVR source

```qvr
# Weibull Parametric Survival Regression
#
# A parametric accelerated-failure-time / proportional-hazards
# survival model with a Weibull baseline. Each item carries a
# covariate whose linear contribution scales the Weibull rate;
# the shape parameter k governs whether the hazard is
# decreasing (k < 1), constant (k = 1, exponential), or
# increasing (k > 1).
#
# Generative structure:
#
#   alpha   ~ Normal(0, 5)                        intercept
#   beta    ~ Normal(0, 5)                        slope
#   k       ~ Gamma(2, 1)                         Weibull shape
#   t_i     ~ Weibull(exp(-(alpha + beta * x_i) / k), k)
#
# The scale reparameterization exp(-eta / k) is the Weibull
# proportional-hazards convention so that increasing eta
# increases the hazard. The model produces uncensored event
# times; right-censoring is handled at the inference layer by
# substituting the Weibull survival function for the density on
# censored rows.

object Item : FinSet 64
object Val : Real 1

program survival_weibull : Item -> Val
    sample alpha <- Normal(loc=0.0, scale=5.0)
    sample beta <- Normal(loc=0.0, scale=5.0)
    sample k <- Gamma(concentration=2.0, rate=1.0)

    let eta = alpha + beta * x
    let scale = exp(-eta / k)

    observe t : Item <- Weibull(scale=scale, concentration=k)
    return beta

export survival_weibull
```

## Understanding the model

Survival time is the elapsed time until an event. The hazard describes the
instantaneous event rate conditional on no earlier event. A Weibull shape
below one makes that rate decrease with elapsed time; a shape above one
makes it increase. The predictor shifts the rate, so a positive coefficient
in this parameterization tends to shorten event times.

An exact event time contributes a [density](../getting-started/concepts.md#5-a-distribution-is-more-than-one-sample)
at that time. A [right-censored record](https://mc-stan.org/docs/stan-users-guide/survival.html#weibull-survival-model)
saying the event has not occurred by time `c` contributes `P(T > c)`
[@stanSurvival]. The current example contains only exact times. Censored
records would require a separate log-survival contribution to the likelihood.

## Walkthrough

The identifier `x` is the exogenous covariate: it is never declared inside the program, so the runtime resolves it from the observations dict at trace time. The reparameterization `scale = exp(-eta / k)` is the Weibull [proportional-hazards](https://en.wikipedia.org/wiki/Proportional_hazards_model) convention: positive shifts in `eta = alpha + beta * x` increase the hazard and shorten survival times. The observed event times `t` are uncensored Weibull draws. The current program does not handle right censoring automatically; a censored model must add the log survival function as an explicit score for censored rows.

The program returns `beta`, a scalar real, so the declared codomain is `Val : Real 1`: the value space of what comes back. `Item` names the plate extent of the event times and appears in the signature only as the domain.

## Try it

> The short fits below demonstrate the API. Assess convergence with multiple chains and diagnostics before interpreting a posterior.

### Generating synthetic data

```python
import torch
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/survival_weibull.qvr")
model = prog.morphism

N = 64
true_alpha = 0.5
true_beta = 1.0
true_k = 2.0

x = torch.randn(N)
eta_true = true_alpha + true_beta * x
scale_true = torch.exp(-eta_true / true_k)
t = torch.distributions.Weibull(scale_true, true_k).sample()

observations = {"x": x, "t": t}
x_in = torch.zeros(N, 1)
```

### SVI fit

```python
from quivers.inference import AutoNormalGuide, ELBO, SVI

oracle_nll = float(
    -torch.distributions.Weibull(scale_true, true_k).log_prob(t).mean()
)

torch.manual_seed(1)
guide = AutoNormalGuide(model, observed_names={"x", "t"})
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

N_mcmc = 32
x_mcmc = x[:N_mcmc]
t_mcmc = t[:N_mcmc]
obs_mcmc = {"x": x_mcmc, "t": t_mcmc}
x_in_mcmc = torch.zeros(N_mcmc, 1)

torch.manual_seed(2)
kernel = NUTSKernel(step_size=0.05, max_tree_depth=3, target_accept=0.8)
mc = MCMC(kernel, num_warmup=20, num_samples=20, num_chains=1)
result = mc.run(model, x_in_mcmc, obs_mcmc)

print(f"acceptance:  {float(result.acceptance_rates.mean()):.2f}")
print(f"divergences: {int(result.divergence_counts.sum())}")
```

## Categorical perspective

The model denotes a conditional kernel into the positive reals. At `k = 1`, the Weibull likelihood reduces to an exponential likelihood; other shape values allow increasing or decreasing hazards.

## Sources and further reading

- *Survival Analysis: Techniques for Censored and Truncated Data* [@klein2003survival].
