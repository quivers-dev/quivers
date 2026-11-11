# Bounded-Outcome Regression via Kumaraswamy

## Overview

A regression on a $(0, 1)$-bounded response with the [Kumaraswamy distribution](https://en.wikipedia.org/wiki/Kumaraswamy_distribution) as the likelihood. Unlike the Beta distribution, its CDF and quantile are available in elementary power functions, which can make inverse-CDF sampling inexpensive. The covariate enters through the first shape parameter under a log link, so the fitted response stays inside the unit interval for every value of the predictor.

Generative structure:

$$
\begin{aligned}
b &\sim \mathrm{HalfNormal}(2),\\
\gamma_0 &\sim \mathrm{Normal}(0, 1),\\
\gamma_1 &\sim \mathrm{Normal}(0, 1),\\
a_n &= \exp(\gamma_0 + \gamma_1 x_n),\\
y_n \mid a_n, b &\sim \mathrm{Kumaraswamy}(a_n, b).
\end{aligned}
$$

The Kumaraswamy density is $f(y; a, b) = a b\, y^{a - 1} (1 - y^a)^{b - 1}$ on $y \in (0, 1)$.

## QVR source

```qvr
# Bounded-Outcome Regression via Kumaraswamy
#
# A regression on a (0, 1)-bounded response. The Kumaraswamy
# distribution is a close cousin of the Beta with a closed-form
# CDF and quantile function, which makes it preferable for models
# that need cheap inverse-CDF sampling or quantile-based
# inference. The first shape parameter carries the linear
# predictor through a log link, so a covariate moves the response
# without ever leaving the unit interval; the second shape
# parameter is global and carries a weakly informative
# HalfNormal prior. Uniformity occurs only at a = b = 1; the
# prior does not itself concentrate the likelihood there.
#
# Generative structure:
#
#   b       ~ HalfNormal(2)                      global second shape
#   gamma_0 ~ Normal(0, 1)                       intercept
#   gamma_1 ~ Normal(0, 1)                       slope
#   a_n     = exp(gamma_0 + gamma_1 * x_n)       per-row first shape
#   y_n     ~ Kumaraswamy(a_n, b)
#
# The predictor x is exogenous data supplied at fit time through
# the observations dict. Resp is the plate the responses are
# observed over and Val is the value space of the returned
# per-row shape.
#
# Reference: [Kumaraswamy 1980](https://doi.org/10.1016/0022-1694(80)90036-0).

object Resp : FinSet 64
object Val : Real 1

program kumaraswamy_regression : Resp -> Val
    sample b <- HalfNormal(scale=2.0)
    sample gamma_0 <- Normal(loc=0.0, scale=1.0)
    sample gamma_1 <- Normal(loc=0.0, scale=1.0)

    let eta = gamma_0 + gamma_1 * x
    let a = exp(eta)

    observe y : Resp <- Kumaraswamy(concentration1=a, concentration0=b)
    return a

export kumaraswamy_regression
```

## Understanding the model

The response must lie strictly between zero and one. The two positive shape
parameters determine where mass lies inside that interval; at `a = b = 1`
the distribution is uniform. The predictor changes `a` through an exponential,
so a coefficient describes a change in log shape rather than a direct change
in the response mean.

A quantile function converts a probability level into a response value.
Applying it to a uniform random number generates a draw, which explains the
synthetic-data code below. After fitting, comparing posterior quantiles at
different predictor values can be more interpretable than comparing the raw
[shape parameters](beta-regression.md#understanding-the-model). Exact endpoint observations require separate treatment.

## Walkthrough

`object Resp : FinSet 64` is the plate the responses are observed over, and `object Val : Real 1` is the value space of what the program returns, so the signature reads `Resp -> Val`. The domain names the plate; the codomain names the space the returned value lives in, which for a real scalar is `Real 1`.

`sample b <- HalfNormal(2.0)` gives the global second shape parameter a weakly-informative positive prior. The uniform Kumaraswamy occurs at $a = b = 1$; a half-Normal is not centered specifically on that value. The two coefficients carry unit-Normal priors, `let eta = gamma_0 + gamma_1 * x` binds the linear predictor over the exogenous covariate `x`, and `let a = exp(eta)` maps it into the positive first shape parameter. `return a` exposes the per-row fitted shape. The implied mean is $E[Y_n] = b\,B(1 + 1/a_n, b)$; this parameterization has no standard one-number concentration equal to `1 / (1 + 1 / b)`.

## Try it

> The short fits below demonstrate the API. Assess convergence with multiple chains and diagnostics before interpreting a posterior.


### Generating synthetic data

Pick ground truth for all three latent sites, build the per-row first shape from the covariate, and forward-generate the bounded responses from those same values, so the point the harness scores is self-consistent.

```python
import torch
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/kumaraswamy_bounded_outcome.qvr")
model = prog.morphism

n_resp = 64

true_b = 1.6
true_gamma_0 = 0.3
true_gamma_1 = 0.7

x = torch.randn(n_resp)
a_true = torch.exp(true_gamma_0 + true_gamma_1 * x)
u = torch.rand(n_resp).clamp(1e-6, 1.0 - 1e-6)
y = (1.0 - (1.0 - u) ** (1.0 / true_b)) ** (1.0 / a_true)

observations = {"y": y, "x": x}
x_in = torch.zeros(1, 1)
```

The response is generated through the Kumaraswamy quantile function $Q(u) = (1 - (1 - u)^{1/b})^{1/a}$, the closed form that motivates the family in the first place.

### SVI fit

```python
from quivers.inference import AutoNormalGuide, ELBO, SVI

torch.manual_seed(1)
guide = AutoNormalGuide(model, observed_names={"y", "x"})
optim = torch.optim.Adam(
    list(model.parameters()) + list(guide.parameters()), lr=5e-2,
)
svi = SVI(model, guide, optim, ELBO(num_particles=1))

losses = [svi.step(x_in, observations) for _ in range(300)]

print(f"initial loss: {losses[0]:.2f}")
print(f"final loss:   {losses[-1]:.2f}")
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

Any bounded-outcome regression where the analyst needs efficient quantile evaluation: financial fraction-of-volume forecasting, biology fraction-of-population modeling, fraction-positive A/B testing where the closed-form quantile speeds Bayesian decision-making. Drop-in alternative to Beta regression when downstream operations are quantile-based rather than moment-based.

## Sources and further reading

- *A generalized probability density function for double-bounded random processes* [@kumaraswamy1980bounded].
