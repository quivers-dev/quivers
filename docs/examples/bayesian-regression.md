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

The blocks below simulate from the model, check the compiled log joint against
a hand-written density, and then recover the parameters with SVI and NUTS. The
program reads its data by name, so its input `x_in` is a single dummy row: that
leading axis is a particle axis, and a longer one would score the whole data
set once per row.

### Generating synthetic data

```python
import torch
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/bayesian_regression.qvr")
model = prog.morphism

N = 64
true_sigma = torch.tensor(0.5)
true_beta_0 = torch.tensor(1.5)
true_beta_1 = torch.tensor(2.0)

x = torch.randn(N)
y = torch.distributions.Normal(true_beta_0 + true_beta_1 * x, true_sigma).sample()
observations = {"x": x, "y": y}
x_in = torch.zeros(1, 1)
```

### Checking the log joint

```python
D = torch.distributions
truth = {"sigma": true_sigma, "beta_0": true_beta_0, "beta_1": true_beta_1}
oracle = (
    D.HalfCauchy(2.0).log_prob(true_sigma)
    + D.Normal(0.0, 5.0).log_prob(true_beta_0)
    + D.Normal(0.0, 2.0).log_prob(true_beta_1)
    + D.Normal(true_beta_0 + true_beta_1 * x, true_sigma).log_prob(y).sum()
)
compiled = model.log_joint(x_in, {**truth, **observations}).sum()
print(f"compiled log joint: {float(compiled):.4f}")
print(f"torch log joint:    {float(oracle):.4f}")
assert torch.allclose(compiled, oracle, rtol=1e-5)
```

### SVI fit

A mean-field [`AutoNormalGuide`](../api/inference/guide.md) is fit by
[`SVI`](../api/inference/svi.md). With 64 rows and noise scale 0.5, the coefficients' standard errors are near
`0.5 / sqrt(64) = 0.0625` and the noise scale's near `0.5 / sqrt(128) = 0.044`,
so the tolerances are four standard errors: 0.25 for each coefficient and 0.18
for `sigma`.

```python
from quivers.inference import AutoNormalGuide, ELBO, SVI

tolerance = {"beta_0": 0.25, "beta_1": 0.25, "sigma": 0.18}

torch.manual_seed(1)
guide = AutoNormalGuide(model, observed_names={"x", "y"})
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
```

### NUTS posterior

[`NUTSKernel`](../api/inference/mcmc.md) samples the same posterior over all
64 rows. Its posterior means should meet the same tolerances and agree with
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
