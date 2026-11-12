# 1. Your first model

Bayesian linear regression in [QVR](../../reference/qvr/index.md), the modeling language of quivers: declare the model, simulate data, fit it by [stochastic variational inference](https://pyro.ai/examples/svi_part_i.html) (SVI) [@hoffman2013svi], and summarize the posterior. Each QVR program carries a typed signature, and the compiler checks the program before any tensor is evaluated.

## Installation

```bash
pip install quivers
```

The following block compiles a one-line module and prints the class of the result:

```python
from quivers.dsl import loads

src = """
object A : FinSet 3
morphism f : A -> A [role=latent]
export f
"""
print(type(loads(src)).__name__)   # Program
```

## The model

Let $y_1, \dots, y_n$ be real responses, $x_i$ a scalar predictor for response $i$, $\beta_0$ and $\beta_1$ an intercept and a slope, and $\sigma$ a noise scale:

$$
\beta_0 \sim \mathrm{Normal}(0, 5), \qquad
\beta_1 \sim \mathrm{Normal}(0, 2), \qquad
\sigma \sim \mathrm{HalfNormal}(1), \qquad
y_i \sim \mathrm{Normal}(\beta_0 + \beta_1 x_i, \sigma).
$$

## QVR, PyMC, and NumPyro

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

The QVR declarations, in order:

| Declaration | Meaning |
|---|---|
| `object Item : FinSet 100` | A finite set `Item` of size 100 that indexes the rows of the data. |
| `program regression : Item -> Item` | A probabilistic program with domain `Item` and codomain `Item`. |
| `sample sigma <- HalfNormal(scale=1.0)` | Draw a latent variable, as `pm.HalfNormal(...)` in PyMC or `numpyro.sample(...)` in NumPyro. |
| `let mu = beta_0 + beta_1 * x_design` | A deterministic binding. `x_design` is bound nowhere in the program, so the compiler records it as a *data input*: a name the program reads but does not bind, given at run time with the observations (Stan's `data` block plays the same part). |
| `observe y : Item <- Normal(loc=mu, scale=sigma)` | A vectorized observe statement, one draw for each element of `Item`. At inference time `y` takes its observed value, and the step adds its log likelihood to the program's log joint density. |

## Effect signatures

The compiler infers which of four *effects* a program body performs:

| Effect | Meaning | Performed by |
|---|---|---|
| `Pure` | no draws and no conditioning | `let` only |
| `Sample` | draws latent variables | `sample v <- F(...)` |
| `Score` | conditions on observations | `observe v <- F(...)` |
| `Marginal` | sums out a discrete latent variable | `marginalize z : K <- ...` with an indented body |

An option `[effects=[...]]` on the program declaration states the set, and the compiler rejects a body that performs an effect outside it. With `[effects=[Pure]]` on `regression`, `loads` raises

```text
CompileError: line 4, col 0: program 'regression' is declared as Pure but body uses effects ['Sample', 'Score']
```

and with `[effects=[Sample]]` it reports that the body uses `Score`. Pyro and NumPyro find such a mistake only when the model runs.

## Data inputs and the observations dict

`loads` accepts a program with a free data input, because the compiler cannot know which inference call will give its value. Two later steps must agree on the names:

1. The guide must know which names are given rather than sampled. Pass them as `observed_names` to [`AutoNormalGuide`](../../api/inference/guide.md): both the observed response `y` and the data input `x_design`.
2. Every SVI step and every MCMC run takes an `observations` dict with a tensor for each of those names.

A name missing from `observed_names` or from `observations` raises at guide construction or at run time, before the model uses the value. [Host data](../../guides/inference-foundations.md#host-data-per-row-covariates-and-index-arrays) (the library's term for data inputs) gives the full rules.

## Compile and fit

`loads` returns a [`Program`](../../api/program.md), a PyTorch `nn.Module`. Its `morphism` attribute is the compiled [`MonadicProgram`](../../api/continuous/programs.md), which the inference drivers in `quivers.inference` fit. Any PyTorch optimizer can also train its parameters directly.

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

The loop has Pyro's structure: a [`Guide`](../../api/inference/guide.md) holds the variational family, an [`Objective`](../../api/inference/elbo.md) such as the [evidence lower bound](https://en.wikipedia.org/wiki/Evidence_lower_bound) (ELBO) is the loss, and an [`SVI`](../../api/inference/svi.md) driver takes the gradient steps. `x_tensor` is the program's input. Each row of the input is one independent run of the program, and a program whose observations live on `Item` takes an input with one row ([input rows are independent runs](../../guides/inference-foundations.md#input-rows-are-independent-runs)).

[`AutoNormalGuide`](../../api/inference/guide.md) is a [mean-field](https://en.wikipedia.org/wiki/Variational_Bayesian_methods) Gaussian guide: each latent variable, mapped to an unconstrained scale, gets an independent Normal factor [@blei2017variational]. SVI trains it with the [reparameterization (pathwise) gradient](https://en.wikipedia.org/wiki/Reparameterization_trick) [@kingma2014autoencoding]. A mean-field guide cannot represent posterior correlation between the intercept and the slope. A [full-covariance guide](06-inference-zoo.md#step-2-which-variational-guide) can.

## Inspect the posterior

[`Predictive`](../../api/inference/predictive.md) draws from a trained [`Guide`](../../api/inference/guide.md) or an `MCMCResult` and returns a dict of `(num_samples, ...)` tensors, one for every latent variable and every observed response:

```python
from quivers.inference import Predictive

predictive = Predictive(model, posterior=guide, num_samples=1000)
draws = predictive(x_tensor, {"x_design": x_data})
print("posterior mean beta_0 =", draws["beta_0"].mean().item())
print("posterior mean beta_1 =", draws["beta_1"].mean().item())
print("posterior mean sigma  =", draws["sigma"].mean().item())
print("predictive mean y[0] =", draws["y"][:, 0].mean().item())   # (1000, 100)
```

`Predictive` batches the draws, passes the same data inputs to every replicate, and returns posterior predictive draws of `y` in the same call. After the 50 steps above the guide is far from converged (the posterior mean of `beta_1` is about 0.43 against a true slope of 2.7). Raise the step count to about 2000 for a usable fit.

## Try this

- Replace [`AutoNormalGuide`](../../api/inference/guide.md) with [`AutoMultivariateNormalGuide`](../../api/inference/guide.md), fit for 2000 steps, and compare the posterior correlation of `beta_0` and `beta_1` under the two guides.
- Add `[effects=[Pure]]` to the `program` line and call `loads` again. Compilation fails with the error shown in [Effect signatures](#effect-signatures).

## Next

[Chapter 2](02-glms.md) replaces the Normal likelihood with Bernoulli and Poisson likelihoods: logistic and Poisson regression.
