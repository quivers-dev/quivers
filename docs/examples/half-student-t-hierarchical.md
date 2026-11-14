# Hierarchical regression with a HalfStudentT scale prior

## Overview

A two-level Bayesian linear regression with heavy-tailed [half-Student-t](https://en.wikipedia.org/wiki/Folded-t_and_half-t_distributions) priors on both scale parameters: the residual scale $\sigma$ and the between-group scale $\tau$. Degrees of freedom $\nu$ control the tail weight, and $\nu \to \infty$ recovers the half-Normal. The prior allows substantial between-group variation while regularizing the scale when few groups are observed. [@gelman2006variance]

The Normal distributions below are parameterized by mean and standard deviation. The generative structure is

$$
\begin{aligned}
\sigma &\sim \mathrm{HalfStudentT}(3, 1),\\
\tau &\sim \mathrm{HalfStudentT}(3, 1),\\
\beta_0 &\sim \mathrm{Normal}(0, 5),\\
\beta_1 &\sim \mathrm{Normal}(0, 2),\\
u_g \mid \tau &\sim \mathrm{Normal}(0, \tau),\\
y_n \mid \beta_0, \beta_1, u, \sigma &\sim \mathrm{Normal}(\beta_0 + u_{g(n)} + \beta_1 x_n, \sigma).
\end{aligned}
$$

## QVR source

```qvr
# Hierarchical Regression with HalfStudentT Scale Prior
#
# A two-level Bayesian linear regression with heavy-tailed
# half-Student-t priors on both scale parameters. HalfStudentT(df,
# scale) is the standard weakly-informative prior for hierarchical
# variance parameters: it concentrates mass near zero (favouring
# pooling) while admitting heavy upper tails when the data
# support substantial residual or between-group variation.
#
# Generative structure:
#
#   sigma   ~ HalfStudentT(3, 1)             residual scale
#   tau     ~ HalfStudentT(3, 1)             between-group scale
#   beta_0  ~ Normal(0, 5)                   population intercept
#   beta_1  ~ Normal(0, 2)                   population slope
#   u_g     ~ Normal(0, tau)                 per-group offset
#   y_n     ~ Normal(beta_0 + u_{g(n)} + beta_1 * x_n, sigma)
#
# The predictor x and the per-row group index group_idx are
# exogenous data provided at fit time through the observations
# dict. Resp is the plate the responses are observed over, Group
# is the plate the offsets are allocated over, and Val is the
# value space of the returned residual scale.
#
# Reference: [Gelman 2006](https://doi.org/10.1214/06-BA117A).

object Group : FinSet 12
object Resp : FinSet 120
object Val : Real 1

program hierarchical_regression : Resp -> Val
    sample sigma <- HalfStudentT(df=3.0, scale=1.0)
    sample tau <- HalfStudentT(df=3.0, scale=1.0)
    sample beta_0 <- Normal(loc=0.0, scale=5.0)
    sample beta_1 <- Normal(loc=0.0, scale=2.0)
    sample u : Group <- Normal(loc=0.0, scale=tau)

    let group_effect = u[group_idx]
    let mu = beta_0 + group_effect + beta_1 * x

    observe y : Resp <- Normal(loc=mu, scale=sigma)
    return sigma

export hierarchical_regression
```

## Understanding the model

There are two kinds of spread to estimate. `sigma` describes responses
varying around their group-adjusted mean. `tau` describes group offsets
varying around zero. Sharing `tau` lets groups with limited data borrow
information about the plausible size of an offset, which is [partial pooling](../tutorials/qvr/03-hierarchical.md#eight-schools-centered).

The half-Student-t prior puts more mass on large scale values than a
half-Normal. All observations share the residual scale `sigma`, so
increasing it widens every conditional Normal distribution. A [Student-t
observation family](https://mc-stan.org/docs/stan-users-guide/regression.html#robust-noise-models)
gives the residual distribution heavier tails even at a fixed scale
[@stanRegression].

## Walkthrough

`object Group : FinSet 12` and `object Resp : FinSet 120` are the two plates: twelve groups and 120 responses, ten per group. `object Val : Real 1` is the value space of what the program returns, a real scalar, so the signature reads `Resp -> Val` rather than `Resp -> Resp`. The domain names the [plate](https://en.wikipedia.org/wiki/Plate_notation) the responses are observed over. The codomain names the space the returned value lives in.

`sample sigma <- HalfStudentT(df=3.0, scale=1.0)` draws the residual scale under a half-Student-t with $\nu = 3$ degrees of freedom and unit scale. The small $\nu$ permits substantial prior mass on large scales. `sample tau <- HalfStudentT(df=3.0, scale=1.0)` gives the between-group scale the same treatment. `sample u : Group <- Normal(loc=0.0, scale=tau)` allocates one offset per group under that scale, and `let group_effect = u[group_idx]` gathers each row's offset through the exogenous `group_idx` design vector. `beta_0` and `beta_1` carry weakly informative Normal priors. `let mu = ...` binds the linear predictor. The `observe` clause scores the response under a Normal likelihood. `return sigma` exposes the residual scale.

## Try it

The blocks below simulate from the model, check the compiled log joint against
a hand-written density, and then recover the parameters with [SVI](../guides/inference-svi.md) (stochastic variational inference) and [NUTS](../guides/inference-mcmc.md) (No-U-Turn Sampler). The input `x_in` has one row, representing one [run over the full data set](../guides/inference-foundations.md#input-rows-are-independent-runs).

### Generating synthetic data

Pick ground truth for every latent site, then forward-generate the responses from those same values so the point the harness scores is self-consistent. The predictor `x` and the group design `group_idx` are exogenous, and reach the model through the observations mapping alongside the response.

```python
import torch
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/half_student_t_hierarchical.qvr")
model = prog.morphism

n_group, n_resp = 12, 120

true_sigma = torch.tensor(0.4)
true_tau = torch.tensor(0.8)
true_beta_0 = torch.tensor(1.5)
true_beta_1 = torch.tensor(2.0)
true_u = torch.distributions.Normal(0.0, true_tau).sample((n_group,))

group_idx = torch.arange(n_group).repeat(n_resp // n_group)
x = torch.randn(n_resp)
mu = true_beta_0 + true_u[group_idx] + true_beta_1 * x
y = torch.distributions.Normal(mu, true_sigma).sample()

observations = {"y": y, "x": x, "group_idx": group_idx}
x_in = torch.zeros(1, 1)
```

### Checking the log joint

```python
D = torch.distributions
truth = {
    "sigma": true_sigma, "tau": true_tau,
    "beta_0": true_beta_0, "beta_1": true_beta_1, "u": true_u,
}


def half_student_t_log_prob(value, df, scale):
    return torch.log(torch.tensor(2.0)) + D.StudentT(df, 0.0, scale).log_prob(value)


oracle = (
    half_student_t_log_prob(true_sigma, 3.0, 1.0)
    + half_student_t_log_prob(true_tau, 3.0, 1.0)
    + D.Normal(0.0, 5.0).log_prob(true_beta_0)
    + D.Normal(0.0, 2.0).log_prob(true_beta_1)
    + D.Normal(0.0, true_tau).log_prob(true_u).sum()
    + D.Normal(mu, true_sigma).log_prob(y).sum()
)
compiled = model.log_joint(x_in, {**truth, **observations}).sum()
print(f"compiled log joint: {float(compiled):.4f}")
print(f"torch log joint:    {float(oracle):.4f}")
assert torch.allclose(compiled, oracle, rtol=1e-5)
```

### SVI fit

A [mean-field](../guides/inference-svi.md) [`AutoNormalGuide`](../api/inference/guide.md) is fit by
[`SVI`](../api/inference/svi.md). The slope rests on all 120 rows (standard error near 0.04) and the residual
scale on 120 residuals (near 0.03), so both are checked to within 0.15. The
between-group scale `tau` rests on only twelve group offsets, so its standard
error is near 0.17 and its tolerance is 0.6. The intercept `beta_0` and the
offsets `u` trade off against each other, so the identified quantities are the
twelve group intercepts `beta_0 + u_g`. Each rests on ten rows (standard error
near 0.13) and is checked to within 0.5.

```python
from quivers.inference import AutoNormalGuide, ELBO, SVI

tolerance = {"beta_1": 0.15, "sigma": 0.15, "tau": 0.6}

torch.manual_seed(1)
guide = AutoNormalGuide(model, observed_names={"y", "x", "group_idx"})
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

group_svi = torch.stack([d["beta_0"].reshape(()) + d["u"].reshape(-1) for d in draws]).mean(0)
assert (group_svi - (true_beta_0 + true_u)).abs().max() < 0.5
```

### NUTS posterior

[`NUTSKernel`](../api/inference/mcmc.md) samples the same posterior over all
120 rows. Its posterior means should meet the same tolerances and agree with
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
group_nuts = (
    result.samples["beta_0"].reshape(-1, 1) + result.samples["u"].reshape(-1, n_group)
).mean(0)
assert (group_nuts - (true_beta_0 + true_u)).abs().max() < 0.5
print(f"divergences: {int(result.divergence_counts.sum())}")
```

## Use cases

The half-Student-t is one option for a positive scale when a half-Normal tail is too light. `HalfCauchy` is the $\nu = 1$ special case. Here the prior governs a shared scale. The observation likelihood remains Normal.
