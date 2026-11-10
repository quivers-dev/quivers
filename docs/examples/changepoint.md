# Bayesian Change-Point Detection

## Overview

A Bayesian [change-point model](https://doi.org/10.2307/2347570) in the rate of a [Poisson](https://en.wikipedia.org/wiki/Poisson_distribution) count series, with a single switch at an unknown step `tau`. The classical application is the British coal-mining disaster series, whose yearly accident rate drops at a change point in the late nineteenth century [@carlin1992changepoint]. The change point is discrete and uniform over the steps of the series, and the program sums it out exactly, so the likelihood of the two rates is the finite mixture over every possible switch location.

## QVR source

```qvr
# Bayesian Change-Point Detection
#
# A single change point in the rate of a Poisson count series, in
# the form analyzed by Carlin, Gelfand, and Smith for the British
# coal-mining disaster counts. The change point is discrete: tau is
# the last step of the first regime, uniform over the Step indices,
# and it is summed out exactly.
#
# Generative structure:
#
#   tau          ~ Uniform{0, ..., T - 1}         last pre-change step
#   rate_before  ~ Gamma(2, 1)                    rate at steps t <= tau
#   rate_after   ~ Gamma(2, 1)                    rate at steps t > tau
#   y_t          ~ Poisson(rate_before)  if t <= tau
#                  Poisson(rate_after)   if t >  tau
#
# The marginalize block enumerates every tau in Tau, scores the whole
# series under that switch, and log-sum-exps the results against the
# uniform prior:
#
#   p(y | rates) = sum_tau p(tau) prod_t Poisson(y_t | rate_t(tau)).
#
# The hierarchical model of Carlin et al. also places hyperpriors on
# the Gamma hyperparameters; this program fixes them at (2, 1).
#
# Reference: [Carlin, Gelfand, and Smith 1992](https://doi.org/10.2307/2347570).

object Step : FinSet 64
object Tau : FinSet 64
object Val : Real 1

program changepoint : Step -> Val
    sample rate_before <- Gamma(concentration=2.0, rate=1.0)
    sample rate_after <- Gamma(concentration=2.0, rate=1.0)

    let tau_prior = factor k : Tau in 1.0 / 64.0

    marginalize tau : Tau <- Categorical(probs=tau_prior) [reduction=logsumexp]
        let after = factor t : Step in real(min(max(t - tau, 0), 1))
        let rate = (1.0 - after) * rate_before + after * rate_after
        observe y : Step <- Poisson(rate=rate)
    return rate_after

export changepoint
```

## Understanding the model

Imagine counts that tend to be high early in a series and lower later. The
unknown `tau` says where the series changes regime: steps up to and including
`tau` use `rate_before`, and later steps use `rate_after`. Because one switch
is shared by the whole series, its likelihood multiplies every row
contribution before [summing over candidates](../tutorials/qvr/04-marginalize.md#hierarchical-mixtures-with-grouping):

$$
p(y \mid r_1, r_2) = \sum_{\tau = 0}^{T - 1} \frac{1}{T}
  \prod_{t \le \tau} \mathrm{Poisson}(y_t \mid r_1)
  \prod_{t > \tau} \mathrm{Poisson}(y_t \mid r_2).
$$

Summing the discrete parameter out leaves a smooth density over the two
rates, which gradient-based inference can explore, while the posterior over
the switch is still recoverable: given the rates, each candidate's
probability is proportional to its term in the sum
[@stanLatentDiscrete]. The value `tau = T - 1` puts every step in the first
regime, so the model also covers a series with no change.

## Walkthrough

`rate_before` and `rate_after` are Gamma-prior intensities, each with prior mean 2 events per step. `tau_prior` is the constant vector `1 / 64` over `Tau`, so `Categorical(probs=tau_prior)` is the uniform prior over the 64 candidate switch steps.

The [`marginalize`](../reference/qvr/probabilistic-programs.md#exact-marginalization) block binds `tau` to each element of `Tau` in turn. Inside it, the [factor expression](../guides/dsl-programs-and-lets.md#factor-expressions-assembling-indexed-tensors) `factor t : Step in real(min(max(t - tau, 0), 1))` builds the regime indicator: it is `0` for `t <= tau` and `1` for `t > tau`, since `t` and `tau` are integers. The per-step rate selects `rate_before` or `rate_after` through that indicator, and `observe y : Step <- Poisson(rate=rate)` scores the whole series under the candidate switch. `[reduction=logsumexp]` then adds the uniform log prior to each candidate's total and log-sum-exps over `Tau`, which is the sum displayed above in log space. `Step` and `Tau` have the same extent because every step is a candidate switch location.

The program returns `rate_after`, a scalar real, so the declared codomain is `Val : Real 1`: the value space of what comes back. `Step` names the plate extent of the observed counts and appears in the signature only as the domain.

## Try it

### Generating synthetic data

Pick ground-truth rates and a switch step, then forward-sample a Poisson count series whose rate drops from `true_rate_before` to `true_rate_after` after step `true_tau`, as in the coal-mining series. The series has exactly the `Step` extent the program declares.

```python
import math

import torch
import torch.distributions as D
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/changepoint.qvr")
model = prog.morphism

T = 64
true_tau = 40
true_rate_before, true_rate_after = 3.0, 1.0
steps = torch.arange(T)
true_rate = torch.where(steps > true_tau, true_rate_after, true_rate_before)
y = torch.poisson(true_rate)

x_in = torch.zeros(1, 1)
observations = {"y": y}
```

### Checking the marginal likelihood

`log_joint` sums `tau` out inside the program. At the true rates it must equal the Gamma prior terms plus the log-sum-exp, written directly in PyTorch, of the 64 per-candidate log likelihoods.

```python
def log_lik_by_tau(rate_before, rate_after, series):
    """Log likelihood of the series under each candidate switch step."""
    t = torch.arange(len(series))
    candidates = torch.arange(len(series)).unsqueeze(1)
    rates = torch.where(t.unsqueeze(0) > candidates, rate_after, rate_before)
    return D.Poisson(rates).log_prob(series).sum(-1)

rate_prior = D.Gamma(2.0, 1.0)
oracle_lj = (
    rate_prior.log_prob(torch.tensor(true_rate_before))
    + rate_prior.log_prob(torch.tensor(true_rate_after))
    + torch.logsumexp(log_lik_by_tau(true_rate_before, true_rate_after, y), 0)
    - math.log(T)
)
model_lj = model.log_joint(
    x_in,
    {
        "rate_before": torch.tensor([true_rate_before]),
        "rate_after": torch.tensor([true_rate_after]),
        "y": y,
    },
)
print(f"model log joint:  {model_lj.item():.4f}")
print(f"oracle log joint: {oracle_lj.item():.4f}")
assert torch.allclose(model_lj, oracle_lj.reshape(1), atol=1e-3)
```

### Exact posterior over the change point

The Gamma prior is conjugate to the Poisson likelihood, so each regime's rate integrates out in closed form and the posterior over `tau` alone is exactly computable. It is the reference the fitted model is compared with below.

```python
def segment_log_marginal(segment, shape=2.0, rate=1.0):
    """log of the integral of Gamma(r | shape, rate) * prod Poisson(segment | r) dr."""
    n, total = len(segment), segment.sum()
    return (
        shape * math.log(rate)
        - math.lgamma(shape)
        + torch.lgamma(shape + total)
        - (shape + total) * math.log(rate + n)
        - torch.lgamma(segment + 1.0).sum()
    )

exact_log_post = torch.stack(
    [segment_log_marginal(y[: k + 1]) + segment_log_marginal(y[k + 1 :]) for k in range(T)]
)
exact_post = torch.softmax(exact_log_post, 0)

order = torch.argsort(exact_post, descending=True)
cumulative = torch.cumsum(exact_post[order], 0)
credible_set = order[: int((cumulative < 0.95).sum()) + 1]
print("posterior mode of tau:", int(exact_post.argmax()))
print("95% credible set:", sorted(credible_set.tolist()))
assert true_tau in credible_set.tolist()
```

### SVI fit

Fit the two rates with a mean-field [`AutoNormalGuide`](../api/inference/guide.md#quivers.inference.guides.AutoNormalGuide); `tau` has no guide site because the program sums it out. The rate tolerances are about three posterior standard deviations, `sqrt(rate / n)` for a regime of `n` steps: `0.8` for the 41 pre-change steps and `0.6` for the 23 post-change steps. The posterior over `tau` implied by the fit averages each candidate's conditional probability over guide draws of the rates, and it should agree with the exact posterior up to a [total variation distance](https://en.wikipedia.org/wiki/Total_variation_distance_of_probability_measures) of `0.1`.

```python
from quivers.inference import AutoNormalGuide, ELBO, SVI

torch.manual_seed(0)
guide = AutoNormalGuide(model, observed_names={"y"})
optim = torch.optim.Adam(guide.parameters(), lr=5e-2)
svi = SVI(model, guide, optim, ELBO())
for _ in range(300):
    svi.step(x_in, observations)

with torch.no_grad():
    draws = [guide.rsample(x_in) for _ in range(300)]
rate_before_draws = torch.stack([d["rate_before"].reshape(-1)[0] for d in draws])
rate_after_draws = torch.stack([d["rate_after"].reshape(-1)[0] for d in draws])
print(f"rate_before: {rate_before_draws.mean().item():.3f} (true {true_rate_before})")
print(f"rate_after:  {rate_after_draws.mean().item():.3f} (true {true_rate_after})")
assert abs(rate_before_draws.mean().item() - true_rate_before) < 0.8
assert abs(rate_after_draws.mean().item() - true_rate_after) < 0.6

fitted_post = torch.stack(
    [
        torch.softmax(log_lik_by_tau(before, after, y), 0)
        for before, after in zip(rate_before_draws, rate_after_draws)
    ]
).mean(0)
tv_distance = 0.5 * (fitted_post - exact_post).abs().sum().item()
print("fitted posterior mode of tau:", int(fitted_post.argmax()))
print(f"total variation from the exact posterior: {tv_distance:.4f}")
assert tv_distance < 0.1
```

### NUTS posterior

With `tau` summed out, the two rates are the only latent sites, and [`NUTSKernel`](../api/inference/mcmc.md#quivers.inference.mcmc.NUTSKernel) samples them under the exact marginal likelihood. Every gradient evaluates all 64 candidate switches, so the run is short and its trees shallow; the posterior over two rates is close to Gaussian, and the same rate tolerances apply. A longer, multi-chain run is needed for calibrated intervals.

```python
from quivers.inference import MCMC, NUTSKernel

kernel = NUTSKernel(step_size=0.1, max_tree_depth=2, target_accept=0.8)
mc = MCMC(kernel, num_warmup=40, num_samples=60, num_chains=1, seed=0)
result = mc.run(model, x_in, observations)

nuts_before = result.samples["rate_before"].mean().item()
nuts_after = result.samples["rate_after"].mean().item()
print("acceptance:", round(float(result.acceptance_rates.mean()), 3))
print("divergences:", int(result.divergence_counts.sum()))
print(f"rate_before: {nuts_before:.3f}, rate_after: {nuts_after:.3f}")
assert abs(nuts_before - true_rate_before) < 0.8
assert abs(nuts_after - true_rate_after) < 0.6
```

## Categorical perspective

The model is a Kleisli morphism in the [Giry monad](https://doi.org/10.1007/BFb0092872)'s [Kleisli category](https://ncatlab.org/nlab/show/Kleisli+category) whose denotation is the joint $p(r_1)\, p(r_2) \sum_{\tau} p(\tau) \prod_t \mathrm{Poisson}(y_t \mid r(t; \tau, r_1, r_2))$. The `marginalize` block is the pushforward of the joint over $\mathrm{Tau} \times$ data along the projection that forgets $\tau$: a finite sum, so the marginal is exact rather than a relaxation of the discrete switch. [@giry1982categorical]

## Sources and further reading

- *Hierarchical Bayesian Analysis of Changepoint Problems* [@carlin1992changepoint].
- *Latent Discrete Parameters*, Stan User's Guide [@stanLatentDiscrete].
- *A categorical approach to probability theory* [@giry1982categorical].
