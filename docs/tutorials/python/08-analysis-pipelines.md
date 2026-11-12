# 8. Analysis pipelines

`quivers.formulas` compiles a regression formula in the notation of lme4 and brms [@bates2015lme4; @burkner2017brms] and a dataframe into a QVR program, and fits it. `y ~ x` models the response column `y` by an intercept and the predictor `x`, and `(1 | g)` adds an intercept for each level of `g`, drawn from a shared population distribution so that groups with little data are pulled toward the population mean (partial pooling, as in [QVR chapter 3](../qvr/03-hierarchical.md)) [@gelman2013bayesian]. Inspect the generated programs, fit Gaussian, hierarchical, ordinal, categorical, and mixture regressions, compare two models by PSIS-LOO, and run a posterior predictive check.

The main objects:

- `fit` parses the formula against a dataframe, compiles it to a [`MonadicProgram`](../../api/continuous/programs.md), and runs the No-U-Turn Sampler (NUTS), Hamiltonian Monte Carlo (HMC), or stochastic variational inference (SVI). It returns a [`BayesianFit`](../../api/formulas/fit.md).
- The posterior of a fit is an `MCMCResult` for NUTS and HMC, whose samples have shape `(chain, draw)` for each site, or a fitted [`Guide`](../../api/inference/guide.md) for SVI.
- [`to_datatree`](../../api/diagnostics/arviz_io.md) converts an `MCMCResult` to the [ArviZ](https://python.arviz.org/) `DataTree` that the diagnostics read. `to_datatree_from_svi` converts SVI draws, and `to_datatree_any` accepts either.

## Setup

The workflow needs the `formulas`, `data`, and `diagnostics` extras:

```bash
pip install "quivers[formulas,data,diagnostics]"
```

```python
import numpy as np
import pandas as pd
import torch
from quivers.formulas import fit, formula_to_qvr
from quivers.diagnostics import to_datatree, compare, posterior_predictive_check
```

## A first fit: Gaussian regression in one line

Simulate 200 rows with a known linear relation:

```python
torch.manual_seed(0)
np.random.seed(0)

N = 200
x = np.random.randn(N)
y = 1.0 + 2.0 * x + 0.5 * np.random.randn(N)
df = pd.DataFrame({"y": y, "x": x})
```

Fit `y ~ x` with a Gaussian family by SVI:

```python
result = fit(
    "y ~ x",
    data=df,
    family="gaussian",
    method="svi",
    num_samples=2000,    # under SVI, num_samples is the number of optimization steps
    seed=0,
)
```

Under `method="svi"`, `num_samples` is the number of optimization steps. Under NUTS and HMC it is the number of draws for each chain.

A [`BayesianFit`](../../api/formulas/fit.md) has these fields:

| Field | Contents |
|---|---|
| `.formula` | The parsed [`Formula`](../../api/formulas/formula.md): columns, family, response. |
| `.program` | The compiled [`MonadicProgram`](../../api/continuous/programs.md). |
| `.posterior` | An `MCMCResult` (NUTS or HMC) or a [`Guide`](../../api/inference/guide.md) (SVI). |
| `.observations` | The observations dict used for inference: response values, [data inputs](../../guides/inference-foundations.md#host-data-per-row-covariates-and-index-arrays), and group indices. |

The QVR program that the formula compiled to:

```python
print(result.qvr_source)
```

```text
object Resp : FinSet 200

program model : Resp -> Resp
    sample intercept <- Normal(loc=0.0, scale=5.0)
    sample beta_x <- Normal(loc=0.0, scale=4.883378911472473)
    let beta_x_per_row = beta_x * x
    sample sigma <- HalfCauchy(scale=2.0)
    let eta = intercept + beta_x_per_row
    let mu = eta
    observe y : Resp <- Normal(loc=mu, scale=sigma)
    return y

export model
```

The predictor values do not appear in the program: `x` is a [data input](../qvr/01-first-model.md#data-inputs-and-the-observations-dict), read from the observations at fit time. The default prior on a coefficient, `Normal(0.0, 5.0)`, is divided by the root mean square of its column, which gives `beta_x` its scale of about 4.88. The intercept has no row index because its column is constant.

Emit the source without fitting:

```python
import tempfile
from pathlib import Path

src = formula_to_qvr("y ~ x", data=df)
with tempfile.TemporaryDirectory() as tmp:
    result.dump_qvr(Path(tmp) / "regression.qvr")   # write the same source to disk
```

### Recover the coefficients

Under SVI the posterior is an `AutoNormalGuide`. Compare the means of its draws with the true values (intercept 1.0, slope 2.0):

```python
draws = [result.posterior.rsample(torch.zeros(1, 1)) for _ in range(200)]
ic = torch.stack([d["intercept"] for d in draws])
bx = torch.stack([d["beta_x"] for d in draws])
print(f"intercept (true 1.0): {ic.mean().item():.3f} ± {ic.std().item():.3f}")
print(f"beta_x    (true 2.0): {bx.mean().item():.3f} ± {bx.std().item():.3f}")
```

```text
intercept (true 1.0): 0.933 ± 0.035
beta_x    (true 2.0): 2.031 ± 0.036
```

The `±` values are standard deviations of the guide's draws, the spread of the approximation. Whether that spread is accurate needs a reference, such as a well-mixed NUTS fit of the same model.

## Random effects: hierarchical model under SVI

Eight groups, each with its own intercept offset, and twelve rows per group:

```python
torch.manual_seed(0)
np.random.seed(0)

N_g, N_per = 8, 12
g = np.repeat(np.arange(N_g), N_per)
alpha = np.random.randn(N_g) * 0.6   # group-level offsets
x = np.random.randn(N_g * N_per)
y = 1.5 + 0.7 * x + alpha[g] + 0.4 * np.random.randn(N_g * N_per)

df = pd.DataFrame({"y": y, "x": x, "g": g})
```

Add the group intercept `(1 | g)`:

```python
result = fit(
    "y ~ x + (1 | g)",
    data=df,
    family="gaussian",
    method="svi",
    num_samples=4000,
    seed=0,
)
print(result.qvr_source)
```

```text
object Resp : FinSet 96

object g : FinSet 8

program model : Resp -> Resp
    sample intercept <- Normal(loc=0.0, scale=5.0)
    sample beta_x <- Normal(loc=0.0, scale=5.019989651784464)
    let beta_x_per_row = beta_x * x
    sample sigma_g_Intercept <- HalfNormal(scale=1.0)
    sample z_g_Intercept : g <- Normal(loc=0.0, scale=1.0)
    let alpha_g_per_row = sigma_g_Intercept * z_g_Intercept[g_idx]
    sample sigma <- HalfCauchy(scale=2.0)
    let eta = intercept + beta_x_per_row + alpha_g_per_row
    let mu = eta
    observe y : Resp <- Normal(loc=mu, scale=sigma)
    return y

export model
```

The group intercept becomes a scale `sigma_g_Intercept` with a `HalfNormal` prior, a plate `z_g_Intercept` of standard Normal draws over the levels of `g`, and the term `sigma_g_Intercept * z_g_Intercept[g_idx]` for each row, where `g_idx` is the index map from rows to groups. This is the non-centered parameterization [@papaspiliopoulos2007parametrization]: the offsets are computed from standardized draws rather than drawn as `Normal(0, sigma_g_Intercept)`.

```python
def mean_param(res, name, n=200):
    draws = torch.stack(
        [res.posterior.rsample(torch.zeros(1, 1))[name] for _ in range(n)]
    )
    return draws.mean().item(), draws.std().item()

for name, true in [
    ("intercept", 1.5),
    ("beta_x", 0.7),
    ("sigma_g_Intercept", 0.6),
    ("sigma", 0.4),
]:
    m, s = mean_param(result, name)
    print(f"{name:20s} (true {true:.2f}): {m:.3f} ± {s:.3f}")
```

```text
intercept            (true 1.50): 2.034 ± 0.037
beta_x               (true 0.70): 0.641 ± 0.041
sigma_g_Intercept    (true 0.60): 0.653 ± 0.040
sigma                (true 0.40): 0.418 ± 0.032
```

The intercept is about 0.5 above its true value, far more than its reported spread of 0.04. The eight simulated offsets `alpha` have mean 0.53, and the data identify only the sum of the intercept and the mean offset, $1.5 + 0.53 \approx 2.03$, which is what the fit recovers. The spread of 0.04 may still be too small, since a mean-field guide ignores the posterior correlation between the intercept and the offsets and tends to understate uncertainty [@blei2017variational]. For a richer approximation pass `guide=AutoMultivariateNormalGuide`, and for sampling use `method="nuts"` and check its diagnostics.

### Options: `method`, `guide`, `reparameterize`

- **`method="nuts" | "hmc" | "svi"`**. The default, `"nuts"`, samples the posterior with the No-U-Turn Sampler and returns an `MCMCResult`. `"svi"` fits a [`Guide`](../../api/inference/guide.md) by optimization, which can be faster on large models. Both need diagnostics.
- **`guide=Cls`** (SVI only). The default is [`AutoNormalGuide`](../../api/inference/guide.md), a mean-field Normal. The alternatives include `AutoMultivariateNormalGuide` (full covariance), `AutoLowRankMultivariateNormalGuide` (memory $O(D \cdot \mathit{rank})$), `AutoLaplaceApproximation` (a Normal at the MAP), and `AutoIAFGuide` (an inverse autoregressive flow), described in [QVR chapter 6](../qvr/06-inference-zoo.md#step-2-which-variational-guide). Pass the class, not an instance: `fit(..., guide=AutoMultivariateNormalGuide)`.
- **`reparameterize="noncentered" | "centered"`**. [`fit`](../../api/formulas/fit.md) and [`formula_to_qvr`](../../api/formulas/compile.md) pass it to the formula compiler, and `result.qvr_source` shows the program it produced.

## Ordinal responses with thresholds for each rater

A cumulative (ordinal) model treats ordered categories as intervals of a latent continuous scale cut at ordered thresholds [@mccullagh1980regression]. The family `cumulative` learns the thresholds. With `thresholds_by`, each level of a grouping column gets its own spacing of the thresholds, partially pooled, beside crossed rater and item intercepts:

<!-- python: skip -->
```python
result = fit(
    "rating ~ condition + (1 | rater) + (1 | item)",
    data=ratings,
    family="cumulative",
    thresholds_by="rater",
    method="svi",
    num_samples=4000,
    seed=0,
)
```

The response labels must be consecutive integers starting at zero. The generated program exponentiates log spacings, sums them cumulatively, and centers each rater's thresholds, so the thresholds are ordered in every draw, the rater intercept is a shift of location, and the thresholds describe how each rater uses the scale.

### Training with a neural predictor

A PyTorch module that produces one score for each row can enter the linear predictor additively:

<!-- python: skip -->
```python
net = torch.nn.Linear(feature_count, 1)

result = fit(
    "rating ~ condition + (1 | rater) + (1 | item)",
    data=ratings,
    family="cumulative",
    thresholds_by="rater",
    predictor=net,
    predictor_data=features,
    method="svi",
    num_samples=4000,
    seed=0,
)
```

The optimizer updates the module's parameters together with the program's and the guide's. The generated QVR reads the module's output as the data input `neural_eta`, and the module itself is an [attachment](../qvr/12-structural-autoencoders.md), held in the Python process and not part of the program. A predictor that is trained must thus use SVI. A frozen predictor also works with NUTS or HMC.

## Unordered choices with varying effects

Encode the alternatives of an unordered response as consecutive integers starting at zero. Three choices with an intercept for each participant:

```python
rng = np.random.default_rng(4)
N = 120
participant = np.repeat(np.arange(12), 10)
x = rng.normal(size=N)
logits = np.column_stack([
    np.zeros(N),
    -0.4 + 1.1 * x,
    0.7 - 0.6 * x,
])
prob = np.exp(logits - logits.max(axis=1, keepdims=True))
prob /= prob.sum(axis=1, keepdims=True)
choice = np.array([rng.choice(3, p=row) for row in prob])
choice_data = pd.DataFrame({
    "choice": choice,
    "x": x,
    "participant": participant,
})
```

The family `categorical` takes category zero as the reference and learns one coefficient vector for each other category:

```python
choice_fit = fit(
    "choice ~ x + (1 | participant)",
    data=choice_data,
    family="categorical",
    method="svi",
    num_samples=250,
    seed=0,
)
```

The program builds an `(N, 3)` matrix of logits whose first column is zero and applies a softmax along the categories. Each fixed coefficient and each non-centered group effect has two entries, one for each non-reference category, so every term is a contrast with the reference category and the logits are identified.

A neural classifier may give all three logits or only the two non-reference ones. `fit` converts a full `(N, 3)` output to contrasts with the reference before adding it:

<!-- python: skip -->
```python
classifier = torch.nn.Linear(feature_count, 3)
choice_fit = fit(
    "choice ~ condition + (1 | participant)",
    data=choice_data,
    family="categorical",
    predictor=classifier,
    predictor_data=features,
    method="svi",
    num_samples=3000,
)
```

## A finite mixture of regressions {#a-finite-mixture-regression}

Each observation comes from one of three Gaussian components, and `x` shifts every component's location:

```python
rng = np.random.default_rng(8)
N = 150
x = rng.normal(size=N)
component = rng.choice(3, size=N, p=[0.25, 0.5, 0.25])
offset = np.array([-2.5, 0.0, 2.5])
y = 0.8 * x + offset[component] + rng.normal(scale=0.4, size=N)
mixture_data = pd.DataFrame({"y": y, "x": x})

mixture_fit = fit(
    "y ~ x",
    data=mixture_data,
    family="mixture",
    mixture_components=3,
    method="svi",
    num_samples=250,
    seed=0,
)
```

The formula intercept is the common location. The compiler centers three component offsets around it, maps two free contrasts of the logits and of the offsets through an orthonormal Helmert basis ([contrasts](https://en.wikipedia.org/wiki/Contrast_(statistics))), turns the three logits into mixture weights, and learns one positive scale for each component. `MixtureNormal` sums over the component assignment inside the likelihood, so the guide has only continuous sites. The components are exchangeable (label switching [@stephens2000labels]), so summarize them by quantities invariant to permutation, or by a labeling convention the application motivates. The SVI budgets of 250 steps keep these fits short. Raise them and check diagnostics before reading either posterior.

## Model comparison with PSIS-LOO

Fit a Bernoulli outcome with and without its predictor, with a small $N$ and short chains:

```python
torch.manual_seed(0)
np.random.seed(0)

N = 80
x = np.random.randn(N)
logit = -0.5 + 1.2 * x
y = np.random.binomial(1, 1 / (1 + np.exp(-logit))).astype(float)
df = pd.DataFrame({"y": y, "x": x})

res_with_x = fit(
    "y ~ x", data=df, family="bernoulli", method="nuts",
    num_warmup=100, num_samples=200, num_chains=2, seed=0,
)
res_intercept = fit(
    "y ~ 1", data=df, family="bernoulli", method="nuts",
    num_warmup=100, num_samples=200, num_chains=2, seed=0,
)
```

PSIS-LOO needs the log likelihood of each observation under each posterior draw [@vehtari2017loo]:

```python
y_obs = torch.as_tensor(y, dtype=torch.float32)
x_t = torch.as_tensor(x, dtype=torch.float32)

def per_row_log_lik(res, y_obs, x_t):
    samples = res.posterior.samples
    ic = samples["intercept"]                         # (chain, draw)
    bx = samples.get("beta_x", torch.zeros_like(ic))  # zero if intercept-only
    logits = ic[..., None] + bx[..., None] * x_t
    p = torch.sigmoid(logits)
    return y_obs * torch.log(p + 1e-12) + (1 - y_obs) * torch.log(1 - p + 1e-12)

ll_with_x = per_row_log_lik(res_with_x, y_obs, x_t)
ll_intercept = per_row_log_lik(res_intercept, y_obs, x_t)
```

Convert each posterior to a `DataTree` with an observed data group and a log likelihood group:

```python
idata_with_x = to_datatree(
    res_with_x.posterior,
    observed_data={"y": y_obs},
    log_likelihood={"y": ll_with_x},
)
idata_intercept = to_datatree(
    res_intercept.posterior,
    observed_data={"y": y_obs},
    log_likelihood={"y": ll_intercept},
)
```

[`compare`](../../api/diagnostics/comparison.md) ranks the fits by PSIS-LOO and computes stacking weights [@yao2018stacking]:

```python
table = compare({"with_x": idata_with_x, "intercept_only": idata_intercept}, round_to=2)
print(table.to_string())
```

```text
                rank  elpd_diff   dse  p_worse diag_diff diag_elpd     p   elpd    se  weight
with_x             0       0.00  0.00      NaN                      2.15 -45.38  4.88    0.85
intercept_only     1      -5.32  3.95     0.91   N < 100            1.01 -50.70  3.45    0.15
```

The model with the predictor ranks first and takes most of the stacking weight. Its advantage in `elpd`, 5.32, is about 1.3 standard errors (`dse` 3.95), and `diag_diff` warns that with fewer than 100 observations the normal approximation behind `dse` and `p_worse` is unreliable. The columns are explained in [QVR chapter 8](../qvr/08-diagnostics-and-comparison.md#comparing-models).

## Posterior predictive checks

[`posterior_predictive_check`](../../api/diagnostics/predictive_checks.md) computes the posterior predictive p-value $\Pr(T(y^{\text{rep}}) \ge T(y))$ of a test statistic $T$ [@gelman1996predictive]. Draw replicated responses under each posterior draw of the intercept and slope, and compare their mean with the observed mean:

```python
samples = res_with_x.posterior.samples
ic, bx = samples["intercept"], samples["beta_x"]
p_pp = torch.sigmoid(ic[..., None] + bx[..., None] * x_t)
pp = torch.bernoulli(p_pp).float()         # (chain, draw, N) replicated y's

idata = to_datatree(
    res_with_x.posterior,
    observed_data={"y": y_obs},
    posterior_predictive={"y": pp},
)
ppc = posterior_predictive_check(idata, observed_name="y", statistic="mean")
print(f"observed mean: {float(ppc['observed']):.3f}")
print(f"predictive mean: {ppc['predictive_mean']:.3f}")
print(f"ppp (mean):    {ppc['ppp']:.3f}")
```

```text
observed mean: 0.312
predictive mean: 0.314
ppp (mean):    0.540
```

A p-value near 0.5 puts the observed mean in the middle of the replicated distribution. Values near 0 or 1 suggest that the model misses the statistic. The mean of a Bernoulli outcome is a weak test, since a model with an intercept tends to reproduce it. The built-in statistics are `mean`, `median`, `sd`, `var`, `min`, `max`, `q05`, and `q95`, and a callable is also accepted. `by=` computes the check separately for each level of a named dimension of the data.

## Recovering the formula

A `BayesianFit` keeps the parsed [`Formula`](../../api/formulas/formula.md) beside the compiled program:

```python
print(result.formula.formula)         # original formula string
print(result.formula.response_name)   # "y"
for col in result.formula.fixed_columns:
    print(f"  {col.qvr_name}: term={col.term!r}, is_intercept={col.is_intercept}")
```

```text
y ~ x + (1 | g)
y
  intercept: term='Intercept', is_intercept=True
  x: term='x', is_intercept=False
```

The translation is a [lens](https://ncatlab.org/nlab/show/lens+%28in+computer+science%29) (`FormulaToQVRModule`): a pair of maps in which the forward map emits the QVR module together with a complement, `FormulaData`, holding what the module does not encode (the data arrays, the original column names, display labels, and the formula string). The backward map rebuilds the original `Formula` from the module and the complement, and `lens.backward(module, complement) == formula` holds, comparing every field, data arrays included. The [formulas API reference](../../api/formulas/compile.md) gives the signatures.

## Next

- The [analysis guide](../../guides/analysis-data-and-formulas.md) documents `quivers.formulas`, `quivers.data`, and `quivers.diagnostics`, including the prior override syntax and the [`DatasetSchema`](../../api/data/schema.md) path from a dataframe to a QVR program without a formula.
- [Chapter 5](05-variational-inference.md) builds and fits a `MonadicProgram` without the formula compiler.
- The [DSL guide](../../guides/dsl-overview.md) covers writing `.qvr` programs directly.
