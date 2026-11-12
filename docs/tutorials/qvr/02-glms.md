# 2. Generalized linear models

A [generalized linear model](https://en.wikipedia.org/wiki/Generalized_linear_model) (GLM) passes a linear predictor through an inverse link function and uses the result as the parameter of a response distribution [@nelder1972glm]. Let $x_i$ be the predictors of observation $i$, $\beta$ the coefficients, $g$ the link function, and $\mathcal F$ the response family:

$$
\theta_i = g^{-1}(\beta^\top x_i), \qquad y_i \sim \mathcal{F}(\theta_i).
$$

Linear regression ([chapter 1](01-first-model.md)) is the case with identity link and Normal family. Logistic regression (logit link, Bernoulli family) and Poisson regression (log link, Poisson family) follow [@stanRegression].

## Logistic regression

=== "QVR"

    ```qvr
    object Item : FinSet 200
    program logistic : Item -> Item [effects=[Sample, Score]]
        sample beta_0 <- Normal(loc=0.0, scale=5.0)
        sample beta_1 <- Normal(loc=0.0, scale=2.0)
        let logit = beta_0 + beta_1 * x_design
        let p     = sigmoid(logit)
        observe y : Item <- Bernoulli(probs=p)
        return y

    export logistic
    ```

=== "PyMC"

    <!-- python: skip -->
    ```python
    with pm.Model() as model:
        beta_0 = pm.Normal("beta_0", 0.0, 5.0)
        beta_1 = pm.Normal("beta_1", 0.0, 2.0)
        p      = pm.invlogit(beta_0 + beta_1 * x_data)
        y      = pm.Bernoulli("y", p, observed=y_data)
    ```

=== "NumPyro"

    <!-- python: skip -->
    ```python
    def logistic(x, y=None):
        beta_0 = numpyro.sample("beta_0", dist.Normal(0.0, 5.0))
        beta_1 = numpyro.sample("beta_1", dist.Normal(0.0, 2.0))
        logit  = beta_0 + beta_1 * x
        numpyro.sample("y", dist.Bernoulli(logits=logit), obs=y)
    ```

`Bernoulli` also accepts the log odds directly: `observe y : Item <- Bernoulli(logits=logit)` drops the `sigmoid` step. Scoring from the logits keeps the log density accurate where the linear predictor is large, whereas `sigmoid(logit)` in single precision rounds to one once `logit` exceeds about 17. Which [transpile targets](../../transpile-support.md) accept the `logits` form is listed there.

`let` expressions provide the usual inverse links and nonlinearities, among them:

| Builtin | Domain | Codomain | Use |
|---|---|---|---|
| `sigmoid(x)` | $\mathbb{R}$ | $(0, 1)$ | inverse of the logit link |
| `softmax(x)` | $\mathbb{R}^K$ | simplex on $K$ | inverse link of a categorical response |
| `log_softmax(x)` | $\mathbb{R}^K$ | log of the simplex | the same, in log space |
| `exp(x)` | $\mathbb{R}$ | $\mathbb{R}_{>0}$ | inverse of the log link (Poisson, Gamma) |
| `log(x)` | $\mathbb{R}_{>0}$ | $\mathbb{R}$ | log link |
| `softplus(x)` | $\mathbb{R}$ | $\mathbb{R}_{>0}$ | positive output that grows linearly rather than exponentially |
| `tanh(x)` | $\mathbb{R}$ | $(-1, 1)$ | bounded nonlinearity |
| `relu(x)` | $\mathbb{R}$ | $\mathbb{R}_{\ge 0}$ | rectified linear unit |
| `sum(x)`, `prod(x)`, `logsumexp(x)` | $\mathbb{R}^K$ | $\mathbb{R}$ | reduce the last axis |
| `cumsum(x)` | $\mathbb{R}^K$ | $\mathbb{R}^K$ | cumulative sum along the last axis |

The [checked primitive reference](../../guides/dsl-programs-and-lets.md#checked-primitive-reference) lists the rest. Fit the model as in chapter 1:

```python
import torch
from quivers.dsl import loads
from quivers.inference import AutoNormalGuide, ELBO, SVI

LOGISTIC_SRC = """
object Item : FinSet 200

program logistic : Item -> Item
    sample beta_0 <- Normal(loc=0.0, scale=5.0)
    sample beta_1 <- Normal(loc=0.0, scale=2.0)
    let logit = beta_0 + beta_1 * x_design
    let p     = sigmoid(logit)
    observe y : Item <- Bernoulli(probs=p)
    return y

export logistic
"""

program = loads(LOGISTIC_SRC)
model   = program.morphism

torch.manual_seed(0)
x_data = torch.randn(200)
true_p = torch.sigmoid(0.3 + 1.5 * x_data)
y_data = torch.bernoulli(true_p)

guide = AutoNormalGuide(model, observed_names={"y", "x_design"})
elbo  = ELBO(num_particles=1)
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(guide.parameters()), lr=5e-3,
)
svi = SVI(model, guide, optimizer, elbo)
x_tensor = torch.zeros(1, 1)
observations = {"x_design": x_data, "y": y_data}
for step in range(50):                          # bump to ~3000 for real fits
    svi.step(x_tensor, observations)
```

## Calibration of the posterior mean probability

A classifier is [calibrated](https://en.wikipedia.org/wiki/Calibration_(statistics)) when, among the observations to which it assigns probability near $p$, a share near $p$ has $y = 1$. The block below averages the success probability over 500 guide draws of $(\beta_0, \beta_1)$, bins the observations by that posterior mean, and compares each bin's observed frequency with a central 95% binomial interval at the bin midpoint.

```python
from scipy.stats import binom

# Average the success probability across guide draws of (beta_0, beta_1).
draws = [guide.rsample(x_tensor) for _ in range(500)]
b0 = torch.stack([d["beta_0"] for d in draws]).reshape(-1)   # (500,)
b1 = torch.stack([d["beta_1"] for d in draws]).reshape(-1)   # (500,)
p_hat = torch.sigmoid(b0[:, None] + b1[:, None] * x_data).mean(dim=0)  # (200,)

# Bin and compare the empirical frequency with a central binomial interval
# evaluated at each bin midpoint. This is not a Wilson interval.
bins = torch.linspace(0, 1, 11)
for lo, hi in zip(bins[:-1], bins[1:]):
    mask = (p_hat >= lo) & (p_hat < hi)
    n = int(mask.sum())
    if n == 0:
        continue
    empirical = y_data[mask].float().mean().item()
    midpoint  = (lo + hi).item() / 2
    band_lo, band_hi = binom.interval(0.95, n, midpoint)
    in_band = "ok" if band_lo / n <= empirical <= band_hi / n else "miss"
    print(f"  [{lo:.1f}, {hi:.1f})  n={n:3d}  predicted={midpoint:.2f}"
          f"  observed={empirical:.2f}  [{band_lo/n:.2f}, {band_hi/n:.2f}]  {in_band}")
```

After 50 steps the output is

```text
  [0.3, 0.4)  n=  4  predicted=0.35  observed=0.00  [0.00, 0.75]  ok
  [0.4, 0.5)  n= 66  predicted=0.45  observed=0.21  [0.33, 0.58]  miss
  [0.5, 0.6)  n=116  predicted=0.55  observed=0.67  [0.46, 0.64]  miss
  [0.6, 0.7)  n= 13  predicted=0.65  observed=1.00  [0.38, 0.92]  miss
  [0.7, 0.8)  n=  1  predicted=0.75  observed=1.00  [0.00, 1.00]  ok
```

The guide has not moved far from its initialization, so the predicted probabilities crowd around 0.5 while the observed frequencies spread from 0.2 to 1. With 3000 steps the predictions spread over all ten bins, and nine of the ten observed frequencies fall inside their intervals. This check uses the posterior mean of the success probability only. A [posterior predictive check](08-diagnostics-and-comparison.md) compares the data with replicated data drawn from the fitted model.

## Poisson regression

```qvr
object Item : FinSet 150
program poisson_reg : Item -> Item [effects=[Sample, Score]]
    sample beta_0 <- Normal(loc=0.0, scale=5.0)
    sample beta_1 <- Normal(loc=0.0, scale=2.0)
    let log_rate = beta_0 + beta_1 * x_design
    let rate     = exp(log_rate)
    observe y : Item <- Poisson(rate=rate)
    return y

export poisson_reg
```

The fitting code is the logistic code with this source and count data in place of binary data.

## Common mistakes

1. **A data input missing from `observed_names`.** If the body reads `x_design` and the guide receives `observed_names={"y"}`, guide construction or the first run fails for want of a value. `loads` cannot detect this, because it records free names as [data inputs](01-first-model.md#data-inputs-and-the-observations-dict). Every name that the body reads and no `sample`, `let`, or `marginalize` step binds belongs in `observed_names`.
2. **A data input of the wrong length.** If `x_design` has shape `(200,)` and `Item` has size 100, broadcasting over `Item` fails with a PyTorch shape error at the first run. The compiler does not know the tensor's shape, so compare tensor lengths with the sizes of the declared objects.
3. **Rebinding a name.** A name bound by `sample` cannot be bound again: `let beta_0 = beta_0 + 1` fails with `variable 'beta_0' already bound in program`. Use a fresh name.

## Comparison with Pyro and NumPyro

1. **Plates come from types.** The annotation `y : Item` on the observe statement makes it a [plate](https://en.wikipedia.org/wiki/Plate_notation) over `Item`, so there is no `pyro.plate` block. [Chapter 3](03-hierarchical.md) draws a group effect `v : G <- Normal(loc=0, scale=sigma)` the same way.
2. **`let` does not sample.** It plays the part of `pm.Deterministic` in PyMC and `numpyro.deterministic` in NumPyro.
3. **One file, several entry points.** A `.qvr` file may declare several programs and computations. `Program.entry_points()` lists them, and `export` selects the one exposed as `program.morphism`.

## Try this

- Add an exposure offset to the Poisson model, `let log_rate = beta_0 + beta_1 * x_design + offset`. The compiler records `offset` as a second data input, so add it to `observed_names` and to `observations`.
- Replace the Bernoulli likelihood with `Binomial(total_count=n_trials, probs=p)`, where `n_trials` is a data input.
- Fit the logistic model with the No-U-Turn Sampler (NUTS) instead of stochastic variational inference (SVI), as in [chapter 6](06-inference-zoo.md#step-1-vi-mcmc-or-hybrid).

## Next

[Chapter 3](03-hierarchical.md) fits hierarchical models, where a mean-field guide tends to misfit and NUTS needs a reparameterization.
