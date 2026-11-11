# Hierarchical regression with a HalfStudentT scale prior

## Overview

A two-level Bayesian linear regression with heavy-tailed [half-Student-t](https://en.wikipedia.org/wiki/Folded-t_and_half-t_distributions) priors on both scale parameters: the residual scale $\sigma$ and the between-group scale $\tau$. Degrees of freedom $\nu$ control the tail weight, and $\nu \to \infty$ recovers the half-Normal. The between-group scale is where the prior earns its keep, since a group-level variance is the parameter a light-tailed prior most often over-shrinks.

Generative structure:

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
# exogenous data supplied at fit time through the observations
# dict. Resp is the plate the responses are observed over, Group
# is the plate the offsets are allocated over, and Val is the
# value space of the returned residual scale.
#
# Reference: [Gelman 2006](https://doi.org/10.1214/06-BA117A).

object Group : FinSet 6
object Resp : FinSet 60
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
varying around their group-adjusted mean; `tau` describes group offsets
varying around zero. Sharing `tau` lets groups with limited data borrow
information about the plausible size of an offset, which is [partial pooling](../tutorials/qvr/03-hierarchical.md#eight-schools-centered).

The half-Student-t prior puts more mass on large scale values than a
half-Normal. All observations share the residual scale `sigma`, so
increasing it widens every conditional Normal distribution. A [Student-t
observation family](https://mc-stan.org/docs/stan-users-guide/regression.html#robust-noise-models)
gives the residual distribution heavier tails even at a fixed scale
[@stanRegression].

## Walkthrough

`object Group : FinSet 6` and `object Resp : FinSet 60` are the two plates: six groups and sixty responses. `object Val : Real 1` is the value space of what the program returns, a real scalar, so the signature reads `Resp -> Val` rather than `Resp -> Resp`. The domain names the plate the responses are observed over; the codomain names the space the returned value lives in.

`sample sigma <- HalfStudentT(3.0, 1.0)` draws the residual scale under a half-Student-t with $\nu = 3$ degrees of freedom and unit scale; the small $\nu$ permits substantial prior mass on large scales. `sample tau <- HalfStudentT(3.0, 1.0)` gives the between-group scale the same treatment. `sample u : Group <- Normal(0.0, tau)` allocates one offset per group under that scale, and `let group_effect = u[group_idx]` gathers each row's offset through the exogenous `group_idx` design vector. `beta_0` and `beta_1` carry weakly-informative Normal priors; `let mu = ...` binds the linear predictor; the `observe` clause scores the response under a Normal likelihood. `return sigma` projects onto the residual-scale posterior, the diagnostic the analyst most commonly inspects.

## Try it

> The short fits below demonstrate the API. Assess convergence with multiple chains and diagnostics before interpreting a posterior.

### Generating synthetic data

Pick ground truth for every latent site, then forward-generate the responses from those same values so the point the harness scores is self-consistent. The predictor `x` and the group design `group_idx` are exogenous, and reach the model through the observations dict alongside the response.

```python
import torch
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/half_student_t_hierarchical.qvr")
model = prog.morphism

n_group, n_resp = 6, 60

true_sigma = 0.4
true_tau = 0.8
true_beta_0 = 1.5
true_beta_1 = 2.0
true_u = torch.distributions.Normal(0.0, true_tau).sample((n_group,))

group_idx = torch.arange(n_group).repeat(n_resp // n_group)
x = torch.randn(n_resp)
mu = true_beta_0 + true_u[group_idx] + true_beta_1 * x
y = torch.distributions.Normal(mu, true_sigma).sample()

observations = {"y": y, "x": x, "group_idx": group_idx}
x_in = torch.zeros(1, 1)
```

### SVI fit

Fit the five latent sites by maximizing the ELBO. The negative ELBO measures the variational objective; the oracle negative log-likelihood is a separate comparison that excludes prior and guide terms.

```python
from quivers.inference import AutoNormalGuide, ELBO, SVI

oracle_nll = float(
    -torch.distributions.Normal(mu, true_sigma).log_prob(y).mean()
)

torch.manual_seed(1)
guide = AutoNormalGuide(model, observed_names={"y", "x", "group_idx"})
optim = torch.optim.Adam(
    list(model.parameters()) + list(guide.parameters()), lr=5e-2,
)
svi = SVI(model, guide, optim, ELBO(num_particles=1))

losses = [svi.step(x_in, observations) for _ in range(300)]

print(f"initial loss: {losses[0]:.2f}")
print(f"final loss:   {losses[-1]:.2f}")
print(f"oracle NLL:   {oracle_nll:.2f}")
```

### NUTS posterior

```python
from quivers.inference import MCMC, NUTSKernel

torch.manual_seed(2)
kernel = NUTSKernel(step_size=0.05, max_tree_depth=6, target_accept=0.8)
mc = MCMC(kernel, num_warmup=100, num_samples=100, num_chains=1)
result = mc.run(model, x_in, observations)

print(f"acceptance:  {float(result.acceptance_rates.mean()):.2f}")
print(f"divergences: {int(result.divergence_counts.sum())}")
```

## Use cases

The half-Student-t is one option for a positive scale when a half-Normal tail is too light. `HalfCauchy` is the $\nu = 1$ special case. Here the prior governs a shared scale; the observation likelihood remains Normal.

## Sources and further reading

- *Prior distributions for variance parameters in hierarchical models* [@gelman2006variance].
