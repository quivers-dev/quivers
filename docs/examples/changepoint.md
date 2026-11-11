# Bayesian Change-Point Detection

## Overview

A canonical Bayesian [change-point model](https://doi.org/10.2307/2347570) in the rate of a [Poisson](https://en.wikipedia.org/wiki/Poisson_distribution) series, with a single switch at an unknown time `tau`. The classical application is the British coal-mining disaster series: the rate of accidents per year drops at an estimated change-point in the late nineteenth century. Two rates, one before and one after the switch, are tied together by a soft indicator parameterized as $s(t) = \sigma(k \cdot (t - \tau))$, which is the standard differentiable relaxation of the indicator $\mathbf{1}\{t > \tau\}$ and lets gradient-based variational inference fit the change-point location directly. [@carlin1992changepoint]

## QVR source

```qvr
# Bayesian Change-Point Detection
#
# A canonical Bayesian change-point model with a single switch
# in the rate of a Poisson series. The classical application is
# the coal-mining disaster series: the rate of accidents per
# year drops at an unknown change-point tau.
#
# Generative structure:
#
#   tau          ~ Uniform(0, T)                  change-point time
#   rate_before  ~ Gamma(2, 1)                    rate before tau
#   rate_after   ~ Gamma(2, 1)                    rate after tau
#   y_t          ~ Poisson(rate_t)                rate_t = soft switch
#
# The soft-indicator parameterisation s = sigmoid(20 * (t - tau))
# is a differentiable relaxation of 1{t > tau} that lets
# gradient-based variational inference fit the change-point
# location directly. Hard change-points require integer-valued
# tau and a per-tau marginalize over a Tau-axis.
#
# Reference: [Carlin, Gelfand, and Smith 1992](https://doi.org/10.2307/2347570).

object Step : FinSet 64
object Val : Real 1

program changepoint : Step -> Val
    sample tau <- Uniform(low=0.0, high=100.0)
    sample rate_before <- Gamma(concentration=2.0, rate=1.0)
    sample rate_after <- Gamma(concentration=2.0, rate=1.0)

    let s = sigmoid(20.0 * (t - tau))
    let rate = (1.0 - s) * rate_before + s * rate_after

    observe y : Step <- Poisson(rate=rate)
    return tau

export changepoint
```

## Understanding the model

Imagine counts that tend to be high early in a series and lower later.
The unknown `tau` controls where the model changes between those regimes.
The sigmoid gate makes a weighted transition between the two rates, so
observations near `tau` can use an intermediate rate. This is a smooth-change
model even when the synthetic series uses a hard switch.

A [posterior interval](../tutorials/qvr/01-first-model.md#inspect-the-posterior)
for `tau` describes uncertainty about the transition's location under this
specification. A model with [finitely many hard switch points](https://mc-stan.org/docs/stan-users-guide/latent-discrete.html#marginalizing-out-the-discrete-parameter)
assigns a posterior probability to each candidate [@stanLatentDiscrete].
Because one switch is shared across the whole series, its likelihood
multiplies all row contributions before [summing over candidates](../tutorials/qvr/04-marginalize.md#hierarchical-mixtures-with-grouping).

## Walkthrough

`tau` is the change-point time with a uniform prior over the 100-unit time window that the 64 steps of the `Step` plate fall inside. `rate_before` and `rate_after` are Gamma-prior intensities, each centered on a prior mean of 2 events per period. The identifier `t` is exogenous host-data: it is never declared inside the program, so the runtime resolves it from the observations dict at trace time, where the caller supplies the per-step vector of absolute observation epochs. The soft indicator $s = \sigma(20 (t - \tau))$ with steepness 20 gives a sharp but differentiable switch; the per-step rate is the convex combination of the two regime rates, and the observation `y : Step <- Poisson(rate)` is the per-step Poisson likelihood.

For a strict hard change-point with integer-valued `tau`, replace the soft indicator with a scoped `marginalize tau_idx : Tau <- Categorical(probs=uniform) [reduction=logsumexp]` block whose indented body emits the per-step likelihood. The runtime then log-sum-exps the per-step likelihood over the `Tau` axis, realizing the exact Bayesian change-point posterior at the cost of a $|\mathsf{Tau}|$-fold widening of the inner loop.

The program returns `tau`, a scalar real, so the declared codomain is `Val : Real 1`: the value space of what comes back. `Step` names the plate extent of the observed counts and appears in the signature only as the domain.

## Try it

> The short fits below demonstrate the API. Assess convergence with multiple chains and diagnostics before interpreting a posterior.

### Generating synthetic data

Pick concrete ground-truth rates and a change-point location, then forward-sample a Poisson count series whose intensity jumps from `rate_before` to `rate_after` at `tau_true`. The per-step time vector `t` is the host-data the program reads from the observations dict, and it carries *absolute* epochs rather than row positions: each count accumulates over one unit-width period of the 100-unit window the `tau` prior spans, and is timestamped at that period's midpoint, so the series runs 18.5, 19.5, ... rather than 0, 1, .... Only `t - tau` enters the likelihood, so the offset leaves the joint unchanged while keeping the observation epoch distinct from the index of the step that reads it.

```python
import torch
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/changepoint.qvr")
model = prog.morphism

T = 64
t_start = 18.5
n_before = 32
tau_true = t_start + n_before
rate_before, rate_after = 2.0, 5.0
y = torch.cat([
    torch.poisson(torch.ones(n_before) * rate_before),
    torch.poisson(torch.ones(T - n_before) * rate_after),
])
t_vec = t_start + torch.arange(T, dtype=torch.float32)
x_in = torch.zeros(1, 1)
observations = {"y": y, "t": t_vec}
```

### SVI fit

Re-initialize the program and fit `tau`, `rate_before`, and `rate_after` by maximizing the ELBO. The oracle log-likelihood at the true rates (piecewise-constant indicator at `tau_true`) is reported for reference.

```python
import torch.distributions as D
from quivers.inference import AutoNormalGuide, ELBO, SVI

torch.manual_seed(0)
prog = load("docs/examples/source/changepoint.qvr")
model = prog.morphism

guide = AutoNormalGuide(model, observed_names={"y"})
optim = torch.optim.Adam(
    list(model.parameters()) + list(guide.parameters()), lr=1e-2,
)
svi = SVI(model, guide, optim, ELBO())

loss0 = svi.step(x_in, observations)
losses = [svi.step(x_in, observations) for _ in range(800)]
loss_final = sum(losses[-20:]) / 20.0
rates_true = torch.where(
    t_vec < tau_true, torch.tensor(rate_before), torch.tensor(rate_after),
)
oracle_ll = D.Poisson(rates_true).log_prob(y).sum().item()
print(f"initial ELBO loss: {loss0:.1f}")
print(f"final ELBO loss:   {loss_final:.1f}")
print(f"oracle -log p(y):  {-oracle_ll:.1f}")
```

### NUTS posterior

The model declares explicit `sample` sites for `tau`, `rate_before`, and `rate_after`, so [`NUTSKernel`](../api/inference/mcmc.md#quivers.inference.mcmc.NUTSKernel) samples them directly. Small warmup and sample budgets keep the snippet inside tens of seconds for an API demonstration; longer, diagnosed runs are needed for posterior interpretation.

```python
from quivers.inference import MCMC, NUTSKernel

torch.manual_seed(0)
prog = load("docs/examples/source/changepoint.qvr")
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

The model is a Kleisli morphism in the [Giry monad](https://doi.org/10.1007/BFb0092872)'s [Kleisli category](https://ncatlab.org/nlab/show/Kleisli+category) whose denotation is the joint $p(\tau, r_1, r_2) \prod_t \mathrm{Poisson}(y_t \mid r(t; \tau, r_1, r_2))$. The soft indicator is the [reparameterization gradient](https://arxiv.org/abs/1312.6114)-friendly relaxation of an indicator function; in the hard-change-point variant, the discrete `tau_idx` is integrated out by `marginalize`, realizing the [right Kan extension](https://ncatlab.org/nlab/show/Kan+extension) along the trivial projection from the time axis to a singleton. [@giry1982categorical] [@kingma2014autoencoding]

## Sources and further reading

- *Hierarchical Bayesian Analysis of Changepoint Problems* [@carlin1992changepoint].
- *Auto-Encoding Variational Bayes* [@kingma2014autoencoding].
- *A categorical approach to probability theory* [@giry1982categorical].
