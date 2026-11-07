# 1. Your first model

This chapter develops a Bayesian linear regression in QVR: it defines the model, generates synthetic data, fits the model with variational inference, and inspects the posterior. Each stage is paired with its analogue in another probabilistic-programming language.

If you're coming from Stan, PyMC, NumPyro, or Pyro, the main difference is that
QVR gives each model a typed signature and checks it before evaluating any
tensors. You can still fit the model with familiar SVI and NUTS workflows. The
categorical implementation becomes relevant in chapter 7.

## Prerequisites

```bash
pip install quivers
```

A 30-second smoke check (paste into a Python REPL):

```python
from quivers.dsl import loads

src = """
object A : FinSet 3
morphism f : A -> A [role=latent]
export f
"""
print(type(loads(src)).__name__)   # Program
```

If that prints `Program` without an error, you're set.

## What inference will answer

Imagine measuring a response at several predictor values. The intercept is
the expected response when the predictor is zero; the slope is the change
associated with increasing the predictor by one; the noise scale describes
variation around that line. The data do not determine these quantities exactly.
We thus describe uncertainty over them with distributions.

A *[prior](../../getting-started/concepts.md#4-a-probabilistic-program-describes-how-data-could-arise)*
describes plausible parameter values before seeing these responses. The
*likelihood* scores the responses for a proposed set of values. The
*posterior* combines the two, assigning more mass to parameter values that
explain the data while respecting the prior. A `sample` line specifies part
of this [generative model](https://pyro.ai/examples/intro_long.html#Background:-probabilistic-models)
[@pyroIntroduction]; the fitting algorithm approximates its posterior.

## The model

We model `n` real-valued observations $y_i$ as a noisy linear function of a scalar predictor $x_i$:

$$
\beta_0 \sim \mathrm{Normal}(0, 5), \qquad
\beta_1 \sim \mathrm{Normal}(0, 2), \qquad
\sigma \sim \mathrm{HalfNormal}(1), \qquad
y_i \sim \mathrm{Normal}(\beta_0 + \beta_1 x_i, \sigma).
$$

## QVR vs PyMC vs NumPyro

=== "QVR (`.qvr`)"

    ```qvr
    object Item : FinSet 100

    program regression : Item -> Item
        sample sigma  <- HalfNormal(scale=1.0)
        sample beta_0 <- Normal(loc=0.0, scale=5.0)
        sample beta_1 <- Normal(loc=0.0, scale=2.0)
        let mu = beta_0 + beta_1 * x_design
        observe y : Item <- Normal(loc=mu, scale=sigma)
        return y

    export regression
    ```

=== "PyMC"

    <!-- python: skip -->
    ```python
    import pymc as pm

    with pm.Model() as model:
        sigma  = pm.HalfNormal("sigma", 1.0)
        beta_0 = pm.Normal("beta_0", 0.0, 5.0)
        beta_1 = pm.Normal("beta_1", 0.0, 2.0)
        mu     = beta_0 + beta_1 * x_data
        y      = pm.Normal("y", mu, sigma, observed=y_data)
    ```

=== "NumPyro"

    <!-- python: skip -->
    ```python
    import numpyro
    import numpyro.distributions as dist

    def regression(x, y=None):
        sigma  = numpyro.sample("sigma", dist.HalfNormal(1.0))
        beta_0 = numpyro.sample("beta_0", dist.Normal(0.0, 5.0))
        beta_1 = numpyro.sample("beta_1", dist.Normal(0.0, 2.0))
        mu     = beta_0 + beta_1 * x
        numpyro.sample("y", dist.Normal(mu, sigma), obs=y)
    ```

Reading the QVR line by line:

| Line | What it says |
|---|---|
| `object Item : FinSet 100` | Declare a finite-set index `Item` of size 100: the row dimension of the data. Domain and codomain are typed objects rather than implicit. |
| `program regression : Item -> Item` | A `program` block is the unit of compilation. The effect set is inferred from the body; you can pin it explicitly with `[effects=[Sample, Score]]` if you want a static check that the body uses only those effects. |
| `sample sigma <- HalfNormal(scale=1.0)` | Draw a random variable. Same as PyMC's `pm.HalfNormal(...)` or NumPyro's `numpyro.sample(...)`. |
| `let mu = beta_0 + beta_1 * x_design` | Deterministic let. The compiler records the free name `x_design` as a host-data parameter; the inference call supplies it in the observations mapping. |
| `observe y : Item <- Normal(loc=mu, scale=sigma)` | Vectorized conditioned bind, one draw per element of `Item`. The runtime sets `y` to the observed value at inference time and scores the likelihood. |

If you're coming from Pyro/NumPyro/Stan, the only feature without a direct analogue is the optional `[effects=[...]]` clause: an *effect signature*. By default the compiler infers which effects (`Sample`, `Score`, `Marginal`) the body uses. Pinning the set explicitly turns it into a static promise. If you write `[effects=[Pure]]` and the body contains a `sample` step, the compiler rejects the program at `loads` time.

### What the effect signature buys you

QVR tracks four effects on every program block:

| Effect | Meaning | Triggered by |
|---|---|---|
| `Pure` | no random draws, no conditioning | `let` only |
| `Sample` | draws latents | `sample v <- F(...)` |
| `Score` | conditions on observations | `observe v <- F(...)` |
| `Marginal` | sums over a discrete latent | `marginalize z : K <- ...` followed by an indented body |

The signature is a static promise. If you declare `[effects=[Pure]]` and the body contains a `sample` or `observe` step, the compiler rejects the program at `loads` time with a typed error pointing at the offending line. Pyro and NumPyro discover the same mistake only when they call the model on real data, possibly inside an inner training loop. QVR catches it in the same pass that catches a typo in an object name.

### The observations-dict contract

Notice the free name `x_design` on line 27: it isn't declared anywhere in the model. QVR calls these *host-data* names. At fit time they have to come from somewhere, and that somewhere is the observations dict you pass through SVI or NUTS. The contract has two parts:

1. The guide needs to know which names will be *supplied externally* (not sampled), via [`AutoNormalGuide`](../../api/inference/guide.md)'s `observed_names` argument. Both observed responses (`y`) and host-data covariates (`x_design`) go here.
2. Every SVI step and every MCMC run takes an `observations` dict that supplies tensors for exactly those names.

`loads` records a free host-data name rather than rejecting it: source
compilation cannot know which inference call will supply the data. Guide
construction and execution enforce the next boundary. A mismatch in
`observed_names`, or a missing value in `observations`, thus fails before
the model can use that value, but it is not a source-compiler diagnostic.

## Compile and fit

QVR programs compile to `nn.Module`. You can train them with the inference stack quivers ships (built around stochastic variational inference, [@hoffman2013svi]), or with any PyTorch optimizer if you want to drop to raw gradients.

```python
import torch
from quivers.dsl import loads
from quivers.inference import AutoNormalGuide, ELBO, SVI

REGRESSION_SRC = """
object Item : FinSet 100

program regression : Item -> Item
    sample sigma  <- HalfNormal(scale=1.0)
    sample beta_0 <- Normal(loc=0.0, scale=5.0)
    sample beta_1 <- Normal(loc=0.0, scale=2.0)
    let mu = beta_0 + beta_1 * x_design
    observe y : Item <- Normal(loc=mu, scale=sigma)
    return y

export regression
"""

program = loads(REGRESSION_SRC)
model   = program.morphism                   # the compiled MonadicProgram

# Synthetic data.
torch.manual_seed(0)
x_data = torch.randn(100)
y_data = 1.5 + 2.7 * x_data + 0.3 * torch.randn(100)

# Variational inference. ``x_design`` is a free name in the model body;
# we hand it to the trace via ``observed_names`` and the observations dict.
guide     = AutoNormalGuide(model, observed_names={"y", "x_design"})
elbo      = ELBO(num_particles=1)
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(guide.parameters()), lr=1e-2,
)
svi = SVI(model, guide, optimizer, elbo)

x_tensor = torch.zeros(1, 1)                  # SVI dispatch input (unused)
observations = {"x_design": x_data, "y": y_data}
for step in range(50):                        # bump to ~2000 for real fits
    loss = svi.step(x_tensor, observations)
```

The pattern is identical to Pyro: a [`Guide`](../../api/inference/guide.md) carries the variational family, an [`Objective`](../../api/inference/elbo.md) is the loss, an [`SVI`](../../api/inference/svi.md) driver runs the loop.

The default [`AutoNormalGuide`](../../api/inference/guide.md) is a diagonal-Gaussian variational posterior trained by a pathwise gradient estimator [@kingma2014autoencoding]. Minibatching is available when the model and call site supply batches.

In the fitting loop, the model defines what should be inferred. The guide
stores an adjustable approximation to the unknown parameters' distribution;
the objective scores that approximation; the optimizer changes its parameters.
`AutoNormalGuide` represents the parameters as independent Normal coordinates
on an unconstrained scale. If a higher intercept requires a lower slope to
fit the same data, this guide cannot represent that posterior dependence;
a [full-covariance guide](06-inference-zoo.md#step-2-which-variational-guide) can represent linear correlations.

## Inspect the posterior

[`Predictive`](../../api/inference/predictive.md) is the single entry point for posterior inspection. It accepts either a trained [`Guide`](../../api/inference/guide.md) or an `MCMCResult` and returns a dict of `(num_samples, ...)` tensors for every latent and every observed response:

```python
from quivers.inference import Predictive

predictive = Predictive(model, posterior=guide, num_samples=1000)
draws = predictive(x_tensor, {"x_design": x_data})
print("posterior mean beta_0 =", draws["beta_0"].mean().item())
print("posterior mean beta_1 =", draws["beta_1"].mean().item())
print("posterior mean sigma  =", draws["sigma"].mean().item())
print("predictive mean y[0] =", draws["y"][:, 0].mean().item())   # (1000, 100)
```

Prefer this over a `for _ in range(N): guide.rsample(...)` loop. `Predictive` batches the draws, threads the same host-data dict through every replicate, and produces posterior-predictive samples in the same call.

## What's different from Pyro/NumPyro?

Three things:

1. **Types on the outside, names on the inside.** Every program has a typed signature `dom -> cod`; latents in the body are scoped to that signature. In Pyro/NumPyro, names live in a global trace and types are implicit.
2. **Compile, then fit.** `loads` runs the QVR compiler before training:
   malformed models, source-level type mismatches, and undefined declarations
   produce errors before tensor evaluation. Shapes that depend on host data are
   checked when those tensors cross the runtime boundary.
3. **Effects in the option block.** The `effects = [Sample, Score, Marginal, Pure]` entry on a program declaration is a static promise about what the body does. It's optional but lets the compiler reject programs that, say, try to `observe` inside a `Pure` block.

## Try this

- Add `composition log_prob [level=algebra]` at the top of the file. The enrichment changes how compositions accumulate scalars: likelihood-style values stay finite under very small probabilities; sometimes useful for long sequence models.
- Replace [`AutoNormalGuide`](../../api/inference/guide.md) with [`AutoMultivariateNormalGuide`](../../api/inference/guide.md) and watch the recovered correlation between `beta_0` and `beta_1`.
- Drop the `[effects=[Sample, Score]]` annotation, then add `[effects=[Pure]]` and re-run. The second case fails compilation with a typed error pointing to the `observe`.

## Next

[Chapter 2](02-glms.md) lifts this model to a generalized linear model: logistic and Poisson regression, with link functions and posterior-predictive calibration plots.

## Sources and further reading

- *Auto-Encoding Variational Bayes* [@kingma2014autoencoding].
