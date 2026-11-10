# Multi-output zero-inflated Poisson regression

## Overview

This program implements zero-inflated Poisson (ZIP) regression [@lambert1992zip]. Each count is a structural zero with probability `pi` and otherwise a Poisson draw with rate `rate`, and each of two outputs carries its own logistic regression for `pi` and its own log-linear regression for the rate. A `marginalize` block sums the binary structural-zero indicator out, so the program scores the exact two-component ZIP likelihood rather than sampling the indicator.

## QVR source

```qvr
# Multi-Output Zero-Inflated Poisson Regression
#
# A multi-output zero-inflated Poisson (ZIP) regression. Each
# count is either a structural zero, with probability pi, or a
# Poisson(rate) draw, which may itself be zero. Each output has
# its own logistic regression for pi and its own log-linear
# regression for the rate.
#
# Generative structure:
#
#   alpha_zero_d, beta_zero_d ~ Normal(0, 5)      structural-zero logit
#   alpha_rate_d, beta_rate_d ~ Normal(0, 5)      log rate
#   pi_n   = sigmoid(alpha_zero_{d(n)} + beta_zero_{d(n)} * x_n)
#   rate_n = exp(alpha_rate_{d(n)} + beta_rate_{d(n)} * x_n)
#   z_n    in {0, 1} with P(z_n = 1) = 1 - pi_n  Poisson branch
#   y_n    ~ Poisson(z_n * rate_n)
#
# z_n = 0 is the structural-zero state. The marginalize block sums
# z_n out, giving the ZIP likelihood
#
#   p(y = 0) = pi + (1 - pi) * exp(-rate)
#   p(y = k) = (1 - pi) * Poisson(k | rate),   k > 0
#
# At a marginalize head, a ContinuousBernoulli prior is scored on
# the two atoms {0, 1} with the Bernoulli weights (pi, 1 - pi);
# its value is real-valued, so the gate z * rate needs no
# Bool-to-number conversion.
#
# Reference: [Lambert 1992](https://doi.org/10.1080/00401706.1992.10485228).

object Out : FinSet 2
object Resp : FinSet 400
object Val : Real 1

program zip_regression : Resp -> Val
    sample alpha_zero : Out <- Normal(loc=0.0, scale=5.0)
    sample beta_zero : Out <- Normal(loc=0.0, scale=5.0)
    sample alpha_rate : Out <- Normal(loc=0.0, scale=5.0)
    sample beta_rate : Out <- Normal(loc=0.0, scale=5.0)

    let az = alpha_zero[out_idx]
    let bz = beta_zero[out_idx]
    let ar = alpha_rate[out_idx]
    let br = beta_rate[out_idx]
    let pi_zero = sigmoid(az + bz * x)
    let pi_poisson = 1.0 - pi_zero
    let rate = exp(ar + br * x)

    marginalize z : Resp <- ContinuousBernoulli(probs=pi_poisson)
        let gated_rate = z * rate
        observe y : Resp <- Poisson(rate=gated_rate)

    return beta_rate

export zip_regression
```

## Understanding the model

There are [two ways to observe zero](https://mc-stan.org/docs/stan-users-guide/finite-mixtures.html#zero-inflation):
the row is a structural zero, or it is a Poisson row whose draw happens to be
zero [@stanFiniteMixtures]. With structural-zero probability `pi` and rate
`lambda`, the zero probability is `pi + (1 - pi) * exp(-lambda)`. A positive
count can only come from the Poisson branch, so its probability is
`(1 - pi) * Poisson(k | lambda)`. This is Lambert's parameterization: `pi` is
the probability of the degenerate zero state, and the logistic coefficients
`alpha_zero` and `beta_zero` describe how that probability varies with `x`.

The branch indicator `z` never needs a value. It is 1 for a row drawn from the
Poisson branch and 0 for a structural zero, so its prior probability of 1 is
`1 - pi`. The [reference marginalization](../tutorials/qvr/04-marginalize.md#a-two-component-gaussian-mixture)
evaluates the block body at both atoms and adds the two branch densities, each
weighted by its prior probability.

## Walkthrough

Per-output coefficient plates `alpha_zero`, `beta_zero` carry the
[logit](https://en.wikipedia.org/wiki/Logit)-link structural-zero probability
`pi_zero`, while `alpha_rate`, `beta_rate` carry the log-link rate. The
`marginalize z : Resp` annotation gives each response row its own branch
indicator, with prior probability `pi_poisson = 1 - pi_zero` of the Poisson
branch. The body is evaluated at `z = 0`, where the gated rate is zero and
`Poisson(0)` puts all its mass on `y = 0`, and at `z = 1`, where the rate is
`rate`. The two log densities are combined by log-sum-exp with log weights
`log(pi_zero)` and `log(1 - pi_zero)`. For a positive count the first branch has
log density `-inf`; log-sum-exp discards it and leaves
`log(1 - pi_zero) + log Poisson(y | rate)`, with finite gradients. Writing the
gate as `z * rate` rather than `(1 - s) * rate` with a structural-zero indicator
`s` also lets a transpiler fold the off atom's rate to the literal `0`, so no
target differentiates `0 * rate` at a positive count.

The prior is written `ContinuousBernoulli` rather than `Bernoulli`. At a
`marginalize` head both are scored on the atoms `{0, 1}` with the Bernoulli
weights `(1 - pi_poisson, pi_poisson)`, so the scored density is the same. The
difference is the value type: a `Bernoulli` value is a `Bool`, and the checked
expression language has no conversion from `Bool` to a number, so `z * rate`
would not type-check. A `ContinuousBernoulli` value is real, and on the two
atoms the gate is exactly `0` or `rate`. A forward draw outside a marginalize
scope would sample the continuous relaxation, but nothing in this program draws
the indicator forward.

The program returns `beta_rate`, an `Out`-indexed plate of real scalars, so the declared codomain is `Val : Real 1`: the per-row value space of the returned coefficients. `Resp` names the flattened response plate (200 rows for each of the two outputs) and appears in the signature only as the domain.

## Try it

The blocks below simulate ZIP data, check the compiled marginal against the
closed-form ZIP log likelihood, and then recover the parameters with SVI and
NUTS. The program reads its data by name, so its input `x_in` is a single dummy
row: that leading axis is a particle axis, and a longer one would score the
whole data set once per row.

### Generating synthetic data

```python
import torch
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/zip_regression.qvr")
model = prog.morphism

N, D = 200, 2
ND = N * D
out_idx = torch.arange(D).repeat(N)
x = torch.randn(ND)

true_alpha_zero = torch.tensor([-1.0, 0.0])
true_beta_zero = torch.tensor([0.5, -1.0])
true_alpha_rate = torch.tensor([0.5, 1.0])
true_beta_rate = torch.tensor([1.0, -0.5])

pi_true = torch.sigmoid(true_alpha_zero[out_idx] + true_beta_zero[out_idx] * x)
rate_true = torch.exp(true_alpha_rate[out_idx] + true_beta_rate[out_idx] * x)
poisson_branch = torch.bernoulli(1.0 - pi_true)
y = poisson_branch * torch.poisson(rate_true)

observations = {"x": x, "y": y, "out_idx": out_idx}
x_in = torch.zeros(1, 1)
```

### Checking the marginal

The closed-form ZIP log likelihood treats zeros and positive counts
separately. The compiled log joint at the true coefficients must equal it plus
the Normal priors, and its gradient must be finite even though every positive
count makes the structural-zero branch score `-inf`.

```python
Normal = torch.distributions.Normal
Poisson = torch.distributions.Poisson


def zip_log_lik(pi, rate, y):
    log_pois = Poisson(rate).log_prob(y)
    log_zero = torch.logaddexp(torch.log(pi), torch.log1p(-pi) + log_pois)
    return torch.where(y == 0, log_zero, torch.log1p(-pi) + log_pois)


truth = {
    "alpha_zero": true_alpha_zero,
    "beta_zero": true_beta_zero,
    "alpha_rate": true_alpha_rate,
    "beta_rate": true_beta_rate,
}
leaves = {k: v.clone().requires_grad_(True) for k, v in truth.items()}
compiled = model.log_joint(x_in, {**leaves, **observations}).sum()
oracle = sum(Normal(0.0, 5.0).log_prob(v).sum() for v in truth.values())
oracle = oracle + zip_log_lik(pi_true, rate_true, y).sum()
print(f"compiled log joint: {float(compiled.detach()):.4f}")
print(f"closed-form ZIP:    {float(oracle):.4f}")
print(f"positive counts:    {int((y > 0).sum())} of {ND}")
assert torch.allclose(compiled, oracle, rtol=1e-5)

grads = torch.autograd.grad(compiled, list(leaves.values()))
assert all(torch.isfinite(g).all() for g in grads)
```

### SVI fit

A mean-field [`AutoNormalGuide`](../api/inference/guide.md) is fit by
[`SVI`](../api/inference/svi.md). The rate coefficients are well determined by
the positive counts, with standard errors near 0.1, so they are checked to
within 0.4 of the truth (about four standard errors). The structural-zero
coefficients are identified only through the excess of zeros over the Poisson
prediction, and their standard errors are near 0.3, so they are checked to
within 1.2 on the same four-standard-error scale.

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

assert (post["alpha_rate"] - true_alpha_rate).abs().max() < 0.4
assert (post["beta_rate"] - true_beta_rate).abs().max() < 0.4
assert (post["alpha_zero"] - true_alpha_zero).abs().max() < 1.2
assert (post["beta_zero"] - true_beta_zero).abs().max() < 1.2
```

### NUTS posterior

[`NUTSKernel`](../api/inference/mcmc.md) samples the same marginal posterior
over all 400 rows; no branch indicator is sampled. Its posterior means should meet the
same tolerances against the truth and agree with the SVI posterior means to
within two standard errors.

```python
from quivers.inference import MCMC, NUTSKernel

torch.manual_seed(2)
kernel = NUTSKernel(step_size=0.05, max_tree_depth=4, target_accept=0.8)
mc = MCMC(kernel, num_warmup=100, num_samples=100, num_chains=1)
result = mc.run(model, x_in, observations)

tolerance = {"alpha_rate": 0.4, "beta_rate": 0.4, "alpha_zero": 1.2, "beta_zero": 1.2}
for k, tol in tolerance.items():
    mean = result.samples[k].reshape(-1, D).mean(0)
    print(f"{k:>10}: posterior mean {mean.numpy().round(2)}, truth {truth[k].numpy()}")
    assert (mean - truth[k]).abs().max() < tol
    assert (mean - post[k]).abs().max() < tol / 2
print(f"divergences: {int(result.divergence_counts.sum())}")
```

## Categorical perspective

The marginalized indicator is a two-object coproduct: one summand maps to `Poisson(0)`, the Dirac kernel at zero, and the other to `Poisson(rate)`. The marginal is the convex combination of the two kernels with weights `(pi, 1 - pi)`, which log-sum-exp computes in log space. The family named at the head fixes those weights; the atoms it is scored on are fixed by the marginalization contract.

## Sources and further reading

- *Zero-Inflated Poisson Regression, With an Application to Defects in Manufacturing* [@lambert1992zip].
