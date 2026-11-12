# Quickstart

Fit a [Bayesian linear regression](https://en.wikipedia.org/wiki/Bayesian_linear_regression)
to simulated data and summarize its posterior.

## 1. Install

```bash
pip install quivers
```

[Installation](installation.md) lists the platforms and the optional extras.

## 2. Write the model

The model has an intercept $\beta_0$, a slope $\beta_1$, and a noise scale
$\sigma$, and it draws one response $y_i$ for each of 100 observations with
predictor $x_i$:

$$
\beta_0 \sim \mathrm{Normal}(0, 5), \qquad
\beta_1 \sim \mathrm{Normal}(0, 2), \qquad
\sigma \sim \mathrm{HalfNormal}(1), \qquad
y_i \sim \mathrm{Normal}(\beta_0 + \beta_1 x_i, \sigma).
$$

Save the QVR program below as `regression.qvr`:

```qvr
object Item : FinSet 100

program regression : Item -> Item
    sample sigma  <- HalfNormal(scale=1.0)
    sample beta_0 <- Normal(loc=0.0, scale=5.0)
    sample beta_1 <- Normal(loc=0.0, scale=2.0)
    let mu = beta_0 + beta_1 * x
    observe y : Item <- Normal(loc=mu, scale=sigma)
    return y

export regression
```

`object Item : FinSet 100` declares a finite set with one element per
observation. Each `sample` step draws a latent variable from its prior, and
`let` computes a value from earlier ones without drawing. The `observe` step
draws one response for each element of `Item`, a
[plate](https://en.wikipedia.org/wiki/Plate_notation) of conditionally
independent draws, and scores the observed `y` under that likelihood. The name
`x` is bound nowhere in the program, so it is a
[data input](../reference/qvr/probabilistic-programs.md#host-data), which the
caller gives at run time with the observations.

## 3. Check it

```bash
qvr check regression.qvr
```

This prints `OK regression.qvr`. An undeclared object or a parameter that a
family does not take is reported with its line and column before any tensor is
evaluated. If line 8 read `observe y : Itm <- ...`, the check would report

```text
regression.qvr:8:16: error[compile]: line 8, col 16: undefined object or space 'Itm'
```

## 4. Fit it

Fit the model by [stochastic variational inference (SVI)](https://pyro.ai/examples/svi_part_i.html)
[@hoffman2013svi], which adjusts a *guide*, a family of distributions over the
latent variables, to maximize the
[evidence lower bound (ELBO)](https://en.wikipedia.org/wiki/Evidence_lower_bound)
[@blei2017variational]:

```python
import torch
from quivers.dsl import load
from quivers.inference import AutoNormalGuide, ELBO, SVI

model = load("regression.qvr").morphism

torch.manual_seed(0)
x = torch.randn(100)
y = 1.5 + 2.7 * x + 0.3 * torch.randn(100)
data = {"x": x, "y": y}

guide = AutoNormalGuide(model, observed_names={"x", "y"})
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(guide.parameters()), lr=0.01
)
svi = SVI(model, guide, optimizer, ELBO())

runs = torch.zeros(1, 1)
for step in range(2000):
    loss = svi.step(runs, data)
```

[`load`](../api/dsl/index.md) compiles the file, and `.morphism` is the compiled
program, a [`MonadicProgram`](../api/continuous/programs.md).
[`AutoNormalGuide`](../api/inference/guide.md) is a mean-field Gaussian guide,
with an independent Normal for each latent variable on an unconstrained scale.
Its `observed_names` are the names that `data` gives, which are the observed
responses and the data inputs. The first argument of
[`SVI.step`](../api/inference/svi.md) is the program's input, each row of which
is [one independent run](../guides/inference-foundations.md#input-rows-are-independent-runs)
of the program. This program reads all its data from `data`, so it takes one row.

## 5. Read the posterior

[`Predictive`](../api/inference/predictive.md) draws the latent variables from
the trained guide and runs the program forward:

```python
from quivers.inference import Predictive

predictive = Predictive(model, posterior=guide, num_samples=1000)
draws = predictive(runs, {"x": x})
for name in ("beta_0", "beta_1", "sigma"):
    print(f"{name}: mean {draws[name].mean():.2f}, sd {draws[name].std():.2f}")
```

```text
beta_0: mean 1.55, sd 0.03
beta_1: mean 2.67, sd 0.03
sigma: mean 0.28, sd 0.02
```

The posterior means lie near the values that generated the data (1.5, 2.7, and
0.3). Because `y` is not passed to `Predictive`, `draws["y"]` holds draws from
the [posterior predictive distribution](https://en.wikipedia.org/wiki/Posterior_predictive_distribution)
of the responses.

## Next

- [Your first model](../tutorials/qvr/01-first-model.md): the same regression, compared line by line with PyMC and NumPyro
- [Choosing an inference algorithm](../tutorials/qvr/06-inference-zoo.md): other guides and objectives, and sampling with NUTS
- [Concepts](concepts.md): composition of weighted relations, the choice of `Algebra`, and the steps of a program
- [Your first quiver](../tutorials/python/01-first-quiver.md): building and composing morphisms in Python
- [Stochastic relations](../tutorials/python/02-stochastic-relations.md): Markov kernels between finite sets and conditioning
- [Categorical structures](../guides/categorical.md): monads and Kleisli categories over weighted relations
- [Interactive tools](../guides/repl-and-lsp.md): the REPL, the Jupyter kernel, and the language server
