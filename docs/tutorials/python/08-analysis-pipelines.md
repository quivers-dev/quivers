# Tutorial 8: Analysis Pipelines

The formula compiler turns a regression specification and a dataframe into a QVR program. This chapter inspects that program, fits hierarchical and Bernoulli regressions with SVI and NUTS, compares models with PSIS-LOO, and runs a posterior-predictive check on synthetic data.

The formula `y ~ x` says that a response column `y` is modeled using predictor
`x` and an intercept. Adding `(1 | g)` gives each group its own intercept
offset, with a common population distribution controlling how much those
offsets can vary. A group with little evidence can borrow information from
that shared distribution; this is [partial pooling](../qvr/03-hierarchical.md#eight-schools-centered). Gelman and colleagues
develop hierarchical modeling and posterior predictive checks
[@gelman2013bayesian].

A fit represents uncertainty over coefficients through a posterior.
Drawing coefficients from it gives a collection of possible regression lines. Predicted responses add observation noise to those draws. The
workflow below first inspects that uncertainty, then asks whether replicated
data resemble observations and whether another model predicts them better.

## Concepts

- **Formula**: A brms / lme4-style string like `"y ~ x + (1 | g)"` that names a response, predictors, and random-effect structure.
- **`fit`**: The one-liner that parses the formula against a dataframe, compiles to a QVR [`MonadicProgram`](../../api/continuous/programs.md), and dispatches to NUTS / HMC / SVI.
- **`BayesianFit`**: The returned record. Carries the parsed `Formula`, the compiled program, the posterior (either an `MCMCResult` or a `Guide`), and the inference-time observations dict.
- **Posterior**: For NUTS / HMC, an [`MCMCResult`](../../api/inference/predictive.md) with `(chain, draw)`-shaped sample tensors per site. For SVI, a fitted [`Guide`](../../api/inference/guide.md) from which you draw via `rsample`.
- **`DataTree`**: The [ArviZ 1.x](https://python.arviz.org/) container used by the diagnostics layer. [`to_datatree`](../../api/diagnostics/arviz_io.md) accepts an `MCMCResult`; [`to_datatree_from_svi`](../../api/diagnostics/arviz_io.md) accepts SVI samples, and [`to_datatree_any`](../../api/diagnostics/arviz_io.md) dispatches between supported inputs.

## Setup

The analysis workflow requires the `formulas`, `data`, and `diagnostics` extras:

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

Generate 200 rows of synthetic data with a known linear relationship:

```python
torch.manual_seed(0)
np.random.seed(0)

N = 200
x = np.random.randn(N)
y = 1.0 + 2.0 * x + 0.5 * np.random.randn(N)
df = pd.DataFrame({"y": y, "x": x})
```

Fit `y ~ x` under a Gaussian family using SVI to optimize a variational [`Guide`](../../api/inference/guide.md):

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

`result` is a [`BayesianFit`](../../api/formulas/fit.md): a frozen `dx.Model` with these load-bearing fields:

| Field         | What it holds                                                                 |
|---------------|--------------------------------------------------------------------------------|
| `.formula`    | The parsed [`Formula`](../../api/formulas/formula.md): columns, family, response. |
| `.program`    | The compiled [`MonadicProgram`](../../api/continuous/programs.md).                |
| `.posterior`  | An [`MCMCResult`](../../api/inference/predictive.md) (NUTS / HMC) or a [`Guide`](../../api/inference/guide.md) (SVI). |
| `.observations` | The inference-time observations dict (response values + host-data covariates + plate-index codes). |

Inspect the QVR program the formula compiled to:

```python
print(result.qvr_source)
```

```text
object Resp : FinSet 200
program model : Resp -> Resp
    sample intercept <- Normal(loc=0.0, scale=5.0)
    sample beta_x <- Normal(loc=0.0, scale=5.0)
    let beta_x_per_row = (beta_x * x)
    sample sigma <- HalfCauchy(scale=2.0)
    let eta = (intercept + beta_x_per_row)
    let mu = eta
    observe y : Resp <- Normal(loc=mu, scale=sigma)
    return y

export model
```

Per-row predictor values do not appear in the program: the `(beta_x * x)` term references a free variable `x` that the inference path resolves from the observations dict at fit time (the *host-data channel*; see [`quivers.inference.trace`](../../api/inference/trace.md)). The intercept latent stays unindexed because the constant-1 design column doesn't need a per-row binding.

You can also emit the source without fitting:

```python
src = formula_to_qvr("y ~ x", data=df)
result.dump_qvr("regression.qvr")          # write the same source to disk
```

### Recover the coefficients

Under SVI the posterior is an `AutoNormalGuide`. Draw from it and check the means against the true values (`intercept=1.0`, `beta_x=2.0`):

```python
draws = [result.posterior.rsample(torch.zeros(1, 1)) for _ in range(200)]
ic = torch.stack([d["intercept"] for d in draws])
bx = torch.stack([d["beta_x"] for d in draws])
print(f"intercept (true 1.0): {ic.mean().item():.3f} ± {ic.std().item():.3f}")
print(f"beta_x    (true 2.0): {bx.mean().item():.3f} ± {bx.std().item():.3f}")
```

```text
intercept (true 1.0): 0.927 ± 0.036
beta_x    (true 2.0): 2.023 ± 0.030
```

The printed `±` values are standard deviations of guide draws. They describe the fitted approximation's spread. To assess its accuracy on this synthetic problem, repeat the fit from different initial values and compare its coefficient intervals with a well-mixed NUTS fit.

## Random effects: hierarchical model under SVI

Eight groups with their own intercept offsets, twelve rows per group:

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

The formula adds a `(1 | g)` random intercept:

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
    sample beta_x <- Normal(loc=0.0, scale=5.0)
    let beta_x_per_row = (beta_x * x)
    sample sigma_g_Intercept <- HalfNormal(scale=1.0)
    sample z_g_Intercept : g <- Normal(loc=0.0, scale=1.0)
    let alpha_g_per_row = (sigma_g_Intercept * z_g_Intercept[g_idx])
    sample sigma <- HalfCauchy(scale=2.0)
    let eta = ((intercept + beta_x_per_row) + alpha_g_per_row)
    let mu = eta
    observe y : Resp <- Normal(loc=mu, scale=sigma)
    return y

export model
```

The random-intercept structure expands to a `HalfNormal` scale latent `sigma_g_Intercept`, a *standard-Normal* plate draw `z_g_Intercept` of size `|g|`, and a per-row contribution `sigma_g_Intercept * z_g_Intercept[g_idx]`. This is the non-centered parameterization: `alpha_g[i] = sigma * z[i]` is computed deterministically rather than drawn as `alpha_g[i] ~ Normal(loc=0, scale=sigma)`. Non-centering often improves posterior geometry in weakly informed hierarchical models.

Recovery:

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
intercept            (true 1.50): 1.748 ± 0.035
beta_x               (true 0.70): 0.620 ± 0.038
sigma_g_Intercept    (true 0.60): 0.273 ± 0.014
sigma                (true 0.40): 0.375 ± 0.028
```

This short SVI fit recovers the fixed effects more closely than the group-level scale. Mean-field VI can underestimate posterior uncertainty and distort variance-component estimates. For a richer approximation, pass `guide=AutoMultivariateNormalGuide`; for sampling-based inference, use `method="nuts"` and inspect its diagnostics.

### Knobs: `method`, `guide`, `reparameterize`

The following choices control the inference path:

- **`method="nuts" | "hmc" | "svi"`**. The default `"nuts"` samples the joint posterior with the No-U-Turn extension to HMC and produces an `MCMCResult`. `"svi"` fits a [`Guide`](../../api/inference/guide.md) by optimization and may be more practical for large models. Either method still requires diagnostics.
- **`guide=Cls`** (SVI only). Defaults to [`AutoNormalGuide`](../../api/inference/guide.md) (mean-field diagonal Normal). Other choices: [`AutoMultivariateNormalGuide`](../../api/inference/guide.md) for a full-rank Cholesky; [`AutoLowRankMultivariateNormalGuide`](../../api/inference/guide.md) for `O(D·rank)` memory on high-dimensional models; [`AutoLaplaceApproximation`](../../api/inference/guide.md) for a Gaussian fit at the MAP; [`AutoIAFGuide`](../../api/inference/guide.md) for an inverse-autoregressive normalizing flow. Pass the class, not an instance: `fit(..., guide=AutoMultivariateNormalGuide)`.
- **`reparameterize="noncentered" | "centered"`**. Both [`fit`](../../api/formulas/fit.md) and [`formula_to_qvr`](../../api/formulas/compile.md) pass this option into the formula compiler. Inspect `result.qvr_source` to confirm the fitted program.

## Ordinal responses with participant-specific scale use

An ordinal response needs ordered cutpoints in addition to its linear
predictor. The cumulative family learns those cutpoints rather than
requiring them as external data. The grouped form below gives each
participant a partially pooled spacing vector while retaining crossed
participant and item intercepts:

<!-- python: skip -->
```python
result = fit(
    "rating ~ condition + (1 | participant) + (1 | item)",
    data=ratings,
    family="cumulative",
    thresholds_by="participant",
    method="svi",
    num_samples=4000,
    seed=0,
)
```

Response labels must be contiguous integers beginning at zero. The
generated program exponentiates log spacings, cumulatively sums them,
and centers each participant's cutpoint vector. Thus cutpoints stay
ordered on every draw, the random intercept remains the location effect,
and the thresholds describe differences in how participants use the
response scale.

### Joint training with a neural predictor

Suppose a neural parser produces one grammaticality score per item. Its
score can enter the same mixed model additively on the latent scale:

<!-- python: skip -->
```python
parser = NeuralChartParser(...)

result = fit(
    "rating ~ condition + (1 | participant) + (1 | item)",
    data=ratings,
    family="cumulative",
    thresholds_by="participant",
    predictor=parser,
    predictor_data=sentence_features,
    method="svi",
    num_samples=4000,
    seed=0,
)
```

The optimizer sees the parser parameters alongside the QVR program and
guide parameters. The generated QVR names the parser output as the free
host input `neural_eta`; the Python module itself remains a runtime
attachment. Trainable predictors thus use SVI. A frozen predictor
can also be used with NUTS or HMC.

## Unordered choices with varying effects

For an unordered response, encode the alternatives as consecutive integers
beginning at zero. This synthetic example has three choices and a
participant-specific intercept:

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

The categorical compiler takes category zero as the reference and learns
two coefficient vectors, one for each non-reference category:

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

The generated program builds an `(N, 3)` logit matrix whose first column
is zero, then applies softmax along the category axis. Fixed coefficients
and non-centered random effects have two entries apiece. Thus every term
has the same reference-category interpretation, and the additive logit
direction is identified.

A neural classifier can contribute either all three logits or the two
non-reference logits. `fit` converts a full `(N, 3)` output to reference
contrasts before adding it to the formula predictor:

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

## A finite-mixture regression

Some continuous responses combine several regimes. Here each observation
comes from one of three Gaussian components, while `x` shifts every
component location:

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

The formula intercept is the common location. The compiler centers three
component offsets around it, projects two free logit and offset contrasts
through an orthonormal Helmert basis, turns the three resulting logits into
shared weights, and learns one positive scale per component. `MixtureNormal`
sums over the component assignment inside the likelihood, so the guide
contains only continuous sites. The components remain exchangeable;
inspect permutation-invariant summaries or impose a substantively
motivated labeling convention when interpreting them. The 250-step SVI
budgets keep these tutorial fits quick; increase them and inspect diagnostics
before interpreting either posterior.

## Model comparison with PSIS-LOO

Two models on the same Bernoulli outcome: with and without the predictor. The smaller `N` and short MCMC keeps the example fast.

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

Compute per-row log-likelihoods under each model's posterior (the ArviZ comparison needs them):

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

Wrap each posterior into an ArviZ `DataTree` with the canonical groups:

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

[`compare`](../../api/diagnostics/comparison.md) delegates to ArviZ's PSIS-LOO ranking and stacking weights:

```python
print(compare({"with_x": idata_with_x, "intercept_only": idata_intercept}))
```

```text
                rank  elpd    p  elpd_diff  weight   se  dse  warning
with_x             0 -43.0  0.0        0.0    0.98  4.4  0.0    False
intercept_only     1 -50.0  0.0       -6.0    0.02  3.3  3.4    False
```

In this illustrative output, `with_x` receives most of the stacking weight. Interpret the comparison from the values produced by your run, including the standard errors and Pareto-$k$ warnings.

## Posterior-predictive checks

[`posterior_predictive_check`](../../api/diagnostics/predictive_checks.md) computes the posterior-predictive p-value for a chosen test statistic. Generate replicated responses under each posterior draw of `(intercept, beta_x)` and check whether the observed mean is in the bulk of the replicated distribution:

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
predictive mean: 0.312
ppp (mean):    0.557
```

A posterior-predictive p-value near 0.5 places the observed mean near the middle of the replicated distribution for this statistic. Values near 0 or 1 can indicate that the model does not capture the chosen statistic, though the interpretation depends on how the check was constructed.

Built-in statistics: `mean`, `median`, `sd`, `var`, `min`, `max`, `q05`, `q95`. The `by=` argument splits the calculation along a named dim (e.g. `by="Verb"` for a per-category check); for arbitrary statistics, pass a callable.

## What about the original formula?

`BayesianFit` keeps both the parsed [`Formula`](../../api/formulas/formula.md) and the compiled program. You can round-trip:

```python
print(result.formula.formula)         # original formula string
print(result.formula.response_name)   # "y"
for col in result.formula.fixed_columns:
    print(f"  {col.qvr_name}: term={col.term!r}, is_intercept={col.is_intercept}")
```

The lens behind the scenes (`FormulaToQVRModule`) is a typed [`dx.Lens`](https://github.com/aaronstevenwhite/didactic) whose forward emits `(Module, FormulaData)`. `FormulaData` is the strict subset of the formula that isn't encoded in the QVR program: the per-row data arrays, the original (pre-`_qvr_name`) identifier names, presentation labels, the formula string itself. Calling `lens.backward(module, complement)` decodes the structural skeleton out of the Module and fuses it with the complement to reproduce the original `Formula` byte-for-byte. See the [formulas API reference](../../api/formulas/compile.md) for the full lens signature.

## Next

- The [Analysis Pipelines guide](../../guides/analysis-data-and-formulas.md) documents `quivers.formulas`, `quivers.data`, and `quivers.diagnostics`, including the full prior-override syntax and the [`DatasetSchema`](../../api/data/schema.md) bridge from dataframes to QVR programs without writing a formula.
- The [Variational Inference tutorial](05-variational-inference.md) shows the lower-level path: building a `MonadicProgram` by hand, tracing, and running SVI without the formula compiler.
- The [DSL guide](../../guides/dsl-overview.md) is the right place to go if you want to write or hand-edit `.qvr` programs directly.
