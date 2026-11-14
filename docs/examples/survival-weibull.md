# Weibull Survival Regression

## Overview

A parametric [survival regression](https://en.wikipedia.org/wiki/Survival_analysis) with a [Weibull](https://en.wikipedia.org/wiki/Weibull_distribution) baseline hazard and right-censored follow-up [@klein2003survival]. Each item carries a covariate `x_i`, an observed time `t_i`, and an event indicator. The covariate multiplies the hazard by `exp(alpha + beta * x_i)`, the [proportional-hazards](https://en.wikipedia.org/wiki/Proportional_hazards_model) form, and the shape `k` governs whether the hazard decreases (k < 1), stays constant (k = 1, exponential), or increases (k > 1) with elapsed time. Rows whose event was observed contribute the Weibull density. Censored rows contribute the probability of surviving past the end of follow-up.

## QVR source

```qvr
# Weibull Proportional-Hazards Survival Regression
#
# A parametric survival regression with a Weibull baseline and
# right-censored follow-up. Each item has a covariate x_i, an
# observed time t_i (provided as log_t = log t_i), and an event
# indicator d_i: 1 when the event was observed at t_i, 0 when
# follow-up ended at t_i without it.
#
# Generative structure:
#
#   alpha   ~ Normal(0, 5)                        intercept
#   beta    ~ Normal(0, 5)                        slope
#   k       ~ Gamma(2, 1)                         Weibull shape
#   eta_i   = alpha + beta * x_i
#   T_i     ~ Weibull(scale=exp(-eta_i / k), concentration=k)
#   t_i     = min(T_i, C_i),  d_i = [T_i <= C_i]
#
# The hazard of T_i is h(t) = k * t**(k - 1) * exp(eta_i): a
# Weibull baseline hazard k * t**(k - 1) multiplied by exp(eta_i),
# which is the proportional-hazards form. The cumulative hazard is
# H(t) = t**k * exp(eta_i). An event row contributes the density
# log h(t_i) - H(t_i) and a censored row the log survival
# function -H(t_i), so row i contributes d_i * log h(t_i) - H(t_i).
#
# Because d_i is 0 or 1, that equals
#
#   log Poisson(d_i | H(t_i)) + d_i * log(k / t_i),
#
# the Poisson form of the censored likelihood: the observe step
# scores the first term and the score step adds the second. The
# censoring times C_i are independent of T_i and carry no
# parameters, so they contribute nothing to the posterior.
#
# Reference: [Aitkin and Clayton 1980](https://doi.org/10.2307/2986301).

object Item : FinSet 200
object Val : Real 1

program survival_weibull : Item -> Val
    sample alpha <- Normal(loc=0.0, scale=5.0)
    sample beta <- Normal(loc=0.0, scale=5.0)
    sample k <- Gamma(concentration=2.0, rate=1.0)

    let eta = alpha + beta * x
    let cumulative_hazard = exp(eta + k * log_t)
    observe event : Item <- Poisson(rate=cumulative_hazard)

    let log_hazard_ratio = log(k) - log_t
    score event_hazard = sum(event * log_hazard_ratio)
    return beta

export survival_weibull
```

## Understanding the model

Survival time is the elapsed time until an event. The hazard `h(t)` is the
instantaneous event rate among items with no earlier event, and the cumulative
hazard `H(t)` is its integral. With scale `exp(-eta / k)` and shape `k`, the
Weibull hazard is `h(t) = k * t**(k - 1) * exp(eta)`: a baseline hazard
`k * t**(k - 1)` shared by every item, multiplied by `exp(eta)`. Hazard ratios
between items are thus constant over time, which is the proportional-hazards
property, and `beta` is a log hazard ratio per unit of `x`. A positive `beta`
raises the hazard and shortens survival times. For the Weibull family the same
model can also be read as an accelerated-failure-time model, with log time
coefficient `-beta / k`, but the parameters here are on the hazard scale.

An exact event time contributes a [density](../getting-started/concepts.md#5-a-distribution-is-more-than-one-sample)
at that time, `log h(t) - H(t)` on the log scale. A
[right-censored record](https://mc-stan.org/docs/stan-users-guide/survival.html#weibull-survival-model)
says only that the event had not occurred by time `t`, so it contributes the
survival probability `P(T > t) = exp(-H(t))` [@stanSurvival]. Both cases are
`d * log h(t) - H(t)` with `d` the event indicator.

## Walkthrough

The identifiers `x` and `log_t` are input data: they are never bound inside the
program, so the runtime reads them from the observations mapping. The times enter
on the log scale, so `H(t) = exp(eta + k * log t)` and `log(k / t) = log k -
log t` need no positivity constraint on the data. The event
indicator `event` is observed data too, bound by the `observe` step.

The program scores the censored likelihood through its Poisson form
[@aitkin1980glim]. For an indicator `d` in `{0, 1}`,

```text
log Poisson(d | H) = d * log H - H
d * log h - H      = log Poisson(d | H) + d * log(h / H)
```

and for the Weibull `h(t) / H(t) = k / t`. The `observe event` step scores
`log Poisson(event | H(t))` per row, and the `score event_hazard` step adds
`event * log(k / t)`, which is nonzero only on event rows. The sum is
`d * log h(t) - H(t)` per row: the Weibull log density on event rows and the
log survival function `-H(t)` on censored rows. This is an identity of
likelihoods, not a claim that the indicator is Poisson distributed. The
posterior over `alpha`, `beta`, and `k` is the censored Weibull posterior,
which the Try-it block below checks against the direct formula. A forward run of
the program draws the indicator from the Poisson [kernel](https://en.wikipedia.org/wiki/Markov_kernel) rather than simulating
event and censoring times, so the synthetic data below is generated in torch
from the Weibull model.

The censoring times themselves never enter the program. Under independent,
non-informative censoring their distribution does not depend on `alpha`, `beta`,
or `k`, so it contributes only a constant to the log posterior.

`return beta` exposes the real log hazard ratio for the covariate. `Item` indexes the observed records.

## Try it

The blocks below simulate censored survival data, check the compiled log joint
against a hand-written censored Weibull log likelihood, and recover the
parameters with [SVI](../guides/inference-svi.md) (stochastic variational inference) and [NUTS](../guides/inference-mcmc.md) (No-U-Turn Sampler). The input `x_in` has one row, representing one [run over the full data set](../guides/inference-foundations.md#input-rows-are-independent-runs).

### Generating synthetic data

Event times come from the model. Censoring times come from an independent
Exponential(0.3) follow-up process, which censors roughly a quarter of the rows.

```python
import torch
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/survival_weibull.qvr")
model = prog.morphism

N = 200
true_alpha = torch.tensor(-0.5)
true_beta = torch.tensor(0.8)
true_k = torch.tensor(1.5)

x = torch.randn(N)
eta_true = true_alpha + true_beta * x
event_time = torch.distributions.Weibull(torch.exp(-eta_true / true_k), true_k).sample()
censor_time = torch.distributions.Exponential(0.3).sample((N,))
t = torch.minimum(event_time, censor_time)
event = (event_time <= censor_time).float()

observations = {"x": x, "log_t": torch.log(t), "event": event}
x_in = torch.zeros(1, 1)
print(f"censored rows: {int((1 - event).sum())} of {N}")
```

### Checking the log joint

The censored Weibull log likelihood, written directly in torch, uses the
Weibull density for event rows and the log survival function `-H(t)` for
censored rows. The compiled log joint at the true parameters must equal it plus
the three priors.

```python
D = torch.distributions
weibull_true = D.Weibull(torch.exp(-eta_true / true_k), true_k)
log_survival = -torch.exp(eta_true) * t**true_k
log_lik = torch.where(event > 0, weibull_true.log_prob(t), log_survival).sum()
log_prior = (
    D.Normal(0.0, 5.0).log_prob(true_alpha)
    + D.Normal(0.0, 5.0).log_prob(true_beta)
    + D.Gamma(2.0, 1.0).log_prob(true_k)
)
truth = {"alpha": true_alpha, "beta": true_beta, "k": true_k}
compiled = model.log_joint(x_in, {**truth, **observations}).sum()
print(f"compiled log joint:    {float(compiled):.4f}")
print(f"censored Weibull + prior: {float(log_lik + log_prior):.4f}")
assert torch.allclose(compiled, log_lik + log_prior, rtol=1e-5)

# Treating the censored times as events gives a different density.
naive = weibull_true.log_prob(t).sum() + log_prior
assert abs(float(naive - compiled)) > 1.0
```

### SVI fit

A [mean-field](../guides/inference-svi.md) [`AutoNormalGuide`](../api/inference/guide.md) is fit by
[`SVI`](../api/inference/svi.md). With about 145 observed events, the standard
errors are near 0.1 for `alpha` and `beta` and near 0.1 for `k`, so the check
asks each posterior mean to sit within 0.3 of the truth.

```python
from quivers.inference import AutoNormalGuide, ELBO, SVI

torch.manual_seed(1)
guide = AutoNormalGuide(model, observed_names={"x", "log_t", "event"})
optim = torch.optim.Adam(guide.parameters(), lr=5e-2)
svi = SVI(model, guide, optim, ELBO(num_particles=1))
for _ in range(1500):
    svi.step(x_in, observations)

with torch.no_grad():
    draws = [guide.rsample(x_in) for _ in range(500)]
post = {k_: torch.stack([d[k_] for d in draws]).mean() for k_ in truth}
for k_ in truth:
    print(f"{k_:>5}: posterior mean {float(post[k_]):.2f}, truth {float(truth[k_]):.2f}")
    assert abs(float(post[k_] - truth[k_])) < 0.3
```

### NUTS posterior

[`NUTSKernel`](../api/inference/mcmc.md) samples the same posterior over all
200 rows. Its posterior means should meet the same tolerance and agree with the
SVI means to within 0.15.

```python
from quivers.inference import MCMC, NUTSKernel

torch.manual_seed(2)
kernel = NUTSKernel(step_size=0.05, max_tree_depth=4, target_accept=0.8)
mc = MCMC(kernel, num_warmup=100, num_samples=100, num_chains=1)
result = mc.run(model, x_in, observations)

for k_ in truth:
    mean = float(result.samples[k_].mean())
    print(f"{k_:>5}: posterior mean {mean:.2f}, truth {float(truth[k_]):.2f}")
    assert abs(mean - float(truth[k_])) < 0.3
    assert abs(mean - float(post[k_])) < 0.15
print(f"divergences: {int(result.divergence_counts.sum())}")
```

## Categorical perspective

Each item's event time is a [Kleisli morphism](https://ncatlab.org/nlab/show/Kleisli+category) of the [Giry monad](https://doi.org/10.1007/BFb0092872) from the covariate to the positive reals, the Weibull kernel with hazard `k * t**(k - 1) * exp(eta)`. Right censoring observes that kernel through a coarser map: an event row observes the time itself, while a censored row observes only the event `T > t`, whose probability is the kernel's survival function. The program does not build that coarsened kernel directly. It scores the same likelihood through the Poisson kernel `H(t) |-> Poisson(H(t))` evaluated at the indicator, plus the event-row factor `k / t`. At `k = 1` the kernel is exponential and the hazard is constant in time. [@giry1982categorical]
