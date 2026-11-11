# Bayesian Linear Regression

## QVR source

```qvr
# Bayesian Linear Regression
#
# A two-parameter linear model with a HalfCauchy prior on the
# noise scale and Normal priors on the regression coefficients.
# Exercises the core sample / let / observe pattern that every
# richer model in the gallery composes from.
#
# Generative structure:
#
#   sigma   ~ HalfCauchy(2)              noise scale
#   beta_0  ~ Normal(0, 5)               intercept
#   beta_1  ~ Normal(0, 2)               slope
#   y_n     ~ Normal(beta_0 + beta_1 * x_n, sigma)
#
# The predictor x is exogenous data: an (N,) tensor supplied at
# fit time via the observations dict alongside the response y.
# The runtime's host-data channel binds free variables in let
# expressions from undeclared keys in the observations dict.

object Resp : FinSet 64
object Val : Real 1

program bayesian_regression : Resp -> Val
    sample sigma <- HalfCauchy(scale=2.0)
    sample beta_0 <- Normal(loc=0.0, scale=5.0)
    sample beta_1 <- Normal(loc=0.0, scale=2.0)

    let mu = beta_0 + beta_1 * x

    observe y : Resp <- Normal(loc=mu, scale=sigma)
    return y

export bayesian_regression
```

## Overview

[Bayesian linear regression](https://en.wikipedia.org/wiki/Bayesian_linear_regression) places priors on the coefficients of a linear model and conditions on observed responses:

$$
\sigma \sim \mathrm{HalfCauchy}(2),\quad \beta_0 \sim \mathcal{N}(0, 5),\quad \beta_1 \sim \mathcal{N}(0, 2),\quad y_n \mid x_n \sim \mathcal{N}(\beta_0 + \beta_1\,x_n,\ \sigma).
$$

The predictor $x$ is exogenous host data supplied alongside the response at fit time; the runtime binds free identifiers in `let` expressions from undeclared keys in the observations dict.

## Understanding the model

Suppose `x` is study time and `y` is a test score. The intercept describes
the expected score at zero study time, the slope describes its change per
unit of study time, and `sigma` describes variation around the line. The
model puts distributions on all three because finite data leave them uncertain.

Each [posterior draw](../tutorials/python/05-variational-inference.md#posterior-inference) specifies a possible line and noise level. Comparing
slopes across draws gives uncertainty about the association; generating a
[new response](../tutorials/python/08-analysis-pipelines.md#posterior-predictive-checks) adds observation noise as well. An interval for a future score
is thus different from an interval for the mean line. Neither calculation
alone establishes a causal effect of study time.

## Walkthrough

`object Resp : FinSet 64` declares the response index: a 64 element finite set that the observed response is indexed by, so `y` carries one row per element and has shape `(64,)`. A `FinSet` names the plate, not the values on it; the value type of `y` comes from the family it is observed under, and `Normal` makes those values real. `object Val : Real 1` names the value space the program returns: `return y` hands back a plate of real scalars, so the codomain is the per-row value space `Real 1`, not the index set `Resp`. The signature `program bayesian_regression : Resp -> Val` is thus a [Kleisli morphism](https://ncatlab.org/nlab/show/Kleisli+category) in the [probability monad](https://ncatlab.org/nlab/show/probability+monad) that scores observed responses under the model's joint kernel.

The three `sample` lines draw the prior parameters as global scalars: `sample sigma <- HalfCauchy(2.0)` puts a heavy-tailed positive prior on the noise scale, and `sample beta_0 <- Normal(0.0, 5.0)`, `sample beta_1 <- Normal(0.0, 2.0)` give the intercept and slope independent Gaussian priors with a wider spread on the intercept than on the slope.

`let mu = beta_0 + beta_1 * x` builds the linear predictor. `x` is not declared as a sample site or a program parameter; it is a free identifier resolved at run time from the host-data channel (the `observations` dict).

`observe y : Resp <- Normal(mu, sigma)` scores the observed response under the Gaussian likelihood, accumulating $\log p(y \mid \mu, \sigma)$ into the program's score effect. `return y` projects the program's joint kernel onto the response site, and `export bayesian_regression` makes the program addressable from the loader.

## DSL features

- **`program` keyword**: Declares a probabilistic program with a type signature (`InputType -> OutputType`) and a monadic body.
- **`<-` (bind)**: Samples a random variable from a distribution. Subsequent statements can depend on the sampled value.
- **Distribution constructors**: `Normal`, `HalfCauchy`, `Exponential`, `Beta`, `Categorical`, etc.
- **`let`**: Deterministic computation over random variables. The result is a derived random variable.
- **`observe`**: Scores supplied data, multiplying the unnormalized model weight by the observation likelihood. Inference approximates the normalized posterior.
- **`return`**: Designates the program's output value.
- **Arithmetic on random variables**: `+`, `-`, `*`, `/` are supported directly.

## Try it

> The short fits below demonstrate the API. Assess convergence with multiple chains and diagnostics before interpreting a posterior.


### Generating synthetic data

```python
import torch
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/bayesian_regression.qvr")
model = prog.morphism

N = 64
true_sigma = 0.5
true_beta_0 = 1.5
true_beta_1 = 2.0

x = torch.randn(N)
y = torch.distributions.Normal(true_beta_0 + true_beta_1 * x, true_sigma).sample()
observations = {"x": x, "y": y}
x_in = torch.zeros(1, 1)
```

### SVI fit

```python
from quivers.inference import AutoNormalGuide, ELBO, SVI

oracle_nll = float(
    -torch.distributions.Normal(true_beta_0 + true_beta_1 * x, true_sigma)
    .log_prob(y)
    .mean()
)

torch.manual_seed(1)
guide = AutoNormalGuide(model, observed_names={"x", "y"})
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

# The plate's size is declared in the source, so NUTS conditions on
# the whole data set, as one run of the program.
obs_mcmc = observations
x_in_mcmc = torch.zeros(1, 1)

torch.manual_seed(2)
kernel = NUTSKernel(step_size=0.05, max_tree_depth=3, target_accept=0.8)
mc = MCMC(kernel, num_warmup=20, num_samples=20, num_chains=1)
result = mc.run(model, x_in_mcmc, obs_mcmc)

print(f"acceptance:  {float(result.acceptance_rates.mean()):.2f}")
print(f"divergences: {int(result.divergence_counts.sum())}")
```

## Categorical perspective

A probabilistic program is a Kleisli morphism $A \to TB$ in the probability monad, where $T$ maps a set to its space of distributions. The `<-` bind operator is Kleisli composition: running $f : A \to TB$ and then $g : B \to TC$ yields $(g \circ_K f) : A \to TC$. In this example, the prior (sampling $\beta_0$, $\beta_1$, $\sigma$) composes with the likelihood ($\mathrm{Normal}(\mu, \sigma)$) via monadic bind, producing a joint distribution over parameters and observations. The `observe` statement weights this joint by the supplied observation density. Normalization defines the posterior $P(\theta \mid y) \propto P(y \mid \theta) \cdot P(\theta)$ by Bayes' rule; executing the statement alone does not compute that normalization. Scored computations are thus interpreted as weighted measures, not necessarily normalized probability kernels. Inference algorithms (VI, MCMC) are computational methods for evaluating the resulting integrals.

## Graphical-model view

A probabilistic program is a procedural encoding of a graphical model. Each `<-` statement is a node; each `observe` statement is an observed variable. In this example: $\beta_0$, $\beta_1$, and $\sigma$ are root nodes (no parents), $\mu$ is computed from $\beta_0$, $\beta_1$, and $x$, and $y$ depends on $\mu$ and $\sigma$.

The same model specification can be fitted with SVI, HMC, or NUTS.

## Extensions

Additional predictor variables and coefficients give a multivariable regression. Parametric program templates and plated group effects give hierarchical variants. A [`GP`](../api/continuous/families.md#quivers.continuous.families.ConditionalGaussianProcess) prior on the regression function gives a nonparametric variant.

## See also

- [DSL Guide: Hierarchical Bayesian Models](../guides/programs-hierarchical.md#hierarchical-models-with-parametric-templates) for the plate-draw, parametric-template, and `marginalize` constructs.
