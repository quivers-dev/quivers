# 2. Generalized linear models

Linear regression is the corner case of a generalized linear model [@nelder1972glm] with an identity link and a normal likelihood. Loosening either gives you a much wider family: logistic regression for binary outcomes, Poisson regression for counts, ordinal regression for graded responses. This chapter does the first two end to end in QVR, with calibration checks on the [posterior predictive](../python/08-analysis-pipelines.md#posterior-predictive-checks).

The model template is the same in every case:

$$
\theta_i = g^{-1}(\beta^\top x_i), \qquad y_i \sim \mathcal{F}(\theta_i),
$$

where $g$ is the link function and $\mathcal{F}$ is the response family. We pick $(g^{-1}, \mathcal{F})$ together.

The linear predictor can be any real number, while a probability must lie
between zero and one and a Poisson rate must be positive. An [inverse link](../../examples/beta-regression.md#understanding-the-model)
connects these requirements. At linear predictor zero, `sigmoid(0)` gives
probability `0.5`, while `exp(0)` gives rate `1`. Under a [logistic link](https://mc-stan.org/docs/stan-users-guide/regression.html#logistic-probit-regression.section),
a coefficient increase changes log-odds by a fixed amount [@stanRegression].
Its effect on probability is smaller near zero or one than near the midpoint.

The response family describes remaining variation after that transformation.
A Bernoulli draw is a binary outcome, even though its probability is continuous;
a Poisson draw is a nonnegative count, even though its rate is continuous.

## Logistic regression

Binary response, sigmoid link, Bernoulli likelihood.

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

`Bernoulli` also takes the log-odds directly, so the same likelihood can be written `observe y : Item <- Bernoulli(logits=logit)` without the `sigmoid` let. Scored from its logits, the density keeps its precision where the linear predictor saturates, while `sigmoid(logit)` in single precision rounds to one once `logit` passes about 17. The transpile targets that have a logit-parameterized Bernoulli (Stan's `bernoulli_logit`, Turing's `BernoulliLogit`, PyMC's `logit_p`, and the `logits` keyword of Pyro, NumPyro, and Edward2) render it natively; BUGS, JAGS, Church, Gen, and WebPPL refuse it by name.

QVR's let-arithmetic exposes the standard nonlinearities so you can name the inverse link explicitly. The full builtin set:

| Builtin | Domain | Codomain | Use |
|---|---|---|---|
| `sigmoid(x)` | $\mathbb{R}$ | $(0, 1)$ | logistic inverse link |
| `softmax(x)` | $\mathbb{R}^K$ | simplex on $K$ | categorical inverse link |
| `log_softmax(x)` | $\mathbb{R}^K$ | log-simplex | numerically stable categorical link |
| `exp(x)` | $\mathbb{R}$ | $\mathbb{R}_{>0}$ | log inverse link (Poisson, Gamma) |
| `log(x)` | $\mathbb{R}_{>0}$ | $\mathbb{R}$ | forward log link |
| `softplus(x)` | $\mathbb{R}$ | $\mathbb{R}_{>0}$ | positive output without the exp blowup |
| `tanh(x)` | $\mathbb{R}$ | $(-1, 1)$ | bounded nonlinearity |
| `relu(x)` | $\mathbb{R}$ | $\mathbb{R}_{\ge 0}$ | rectified linear |
| `clamp(x, lo, hi)` | $\mathbb{R}$ | $[\text{lo}, \text{hi}]$ | constrain |
| `sum(x)`, `prod(x)`, `cumsum(x)`, `logsumexp(x)` | tensor | scalar / tensor | reductions over an axis |

(A *link function* is the deterministic map $g$ in $g(\theta_i) = \beta^\top x_i$. We compose with its inverse $g^{-1}$ to turn the linear predictor into the chosen response parameter, such as a probability or mean.)

Fitting is the same as chapter 1:

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

## Posterior-predictive calibration

A logistic regression is well-calibrated if, of all observations where the model predicts $p = 0.7$, roughly 70% actually have $y = 1$.

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

## Poisson regression

Count response, log link, Poisson likelihood.

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

The fit code is identical to the logistic case; only the model file changes.
`Predictive` handles the count-valued likelihood without additional
configuration.

## Common mistakes

Most "why does this fail at runtime" questions trace to one of three slips:

1. **`observed_names` mismatch.** If the body references `x_design` but you pass `observed_names={"y"}`, the inference boundary lacks a host-data value. `loads` records such free names as program inputs; guide construction or the first execution, rather than source compilation, diagnoses a missing runtime value. Rule of thumb: every name that appears in the body but is not introduced by `<-`, `let`, or `marginalize` belongs in `observed_names`.
2. **Shape mismatch in host data.** If `x_design` has shape `(200,)` but the object `Item` was declared `: 100`, the per-row broadcast over `Item` fails with a torch shape error. The compiler doesn't know `x_design`'s runtime shape, so this one surfaces at the first forward pass. Sanity-check tensor lengths against object cardinalities.
3. **Re-using an `<-` name on the LHS of `let`.** Once a name is bound by a sample step, you can't reassign it. To name a deterministic transform, pick a fresh name (`let mu = beta_0 + beta_1 * x`, not `let beta_0 = beta_0 + 1`).

## What QVR adds

Three observations from this chapter you may have already noticed:

1. **No `pyro.plate` / `numpyro.plate` wrapping.** Plates are inferred from object cardinalities and the domain/codomain typing. If you need an explicit indexed family (say a per-group intercept), chapter 3 introduces the plate-draw syntax (`v : G <- Normal(loc=0, scale=sigma)`).
2. **`let` is not sampling.** PyMC `pm.Deterministic`, NumPyro `numpyro.deterministic`, Pyro `pyro.deterministic`: every PPL has a different name for "this is computed from existing variables without a new draw." QVR uses `let`. The compiler tracks the dependency for autograd.
3. **Several entries, one module.** A `.qvr` file may declare several
   programs and computations. `Program.entry_points()` lists the checked
   entry points; `export` selects the classic compiled morphism exposed
   as `program.morphism`.

## Try this

- Add a constant offset (`offset`) to the Poisson model: `let log_rate = beta_0 + beta_1 * x + offset`. The compiler treats `offset` as a free variable and exposes it as a model input alongside `x`.
- Switch the Bernoulli likelihood to a Binomial with a known trial count `n_trials`. The Binomial constructor takes `(n, p)`.
- Run NUTS instead of SVI. Skip ahead to chapter 6, line "Calling NUTS"; it's a five-line change.

## Next

[Chapter 3](03-hierarchical.md) covers hierarchical models: the place where mean-field VI starts to misbehave and where NUTS earns its keep.

## Sources and further reading

- *Generalized Linear Models* [@nelder1972glm].
