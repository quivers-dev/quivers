# Analysis: Fitting and Diagnostics

`fit(...)` estimates the posterior of a model given by a formula, and
`quivers.diagnostics` checks the result with [ArviZ](https://python.arviz.org/) and
with convergence and calibration diagnostics of its own. `quivers.analysis` derives
initializations from a program's structure. [Data and Formulas](analysis-data-and-formulas.md)
covers data loading and formulas.

## One-line fit

A fit combines a model (formula, family, and priors), an inference method, and data.
The result keeps the generated QVR source with the posterior approximation, and the
source shows which parameters were introduced and which data columns they read.

<!-- python: skip -->
```python
from quivers.formulas import fit

result = fit(
    "acceptability ~ verb + frame + log(rt) + (1 + verb | subject)",
    data=df,                       # pandas or polars
    family="bernoulli",            # auto-derives the logit link
    method="nuts",
    num_warmup=500,
    num_samples=1000,
    num_chains=4,
    seed=0,
)
```

The result is a [`BayesianFit`](../api/formulas/fit.md), a frozen
[`dx.Model`](https://didactic.dev/api/Model) with `.formula`, `.family`, `.program`
(the compiled [`MonadicProgram`](../api/continuous/programs.md)), `.posterior` (an
[`MCMCResult`](../api/inference/mcmc.md) or a [`Guide`](../api/inference/guide.md)),
and `.observations` (the observations dict used for inference).

<!-- python: skip -->
```python
print(result.qvr_source)                    # canonical .qvr source
result.dump_qvr("acceptability.qvr")        # write to disk
```

### Prior overrides

<!-- python: skip -->
```python
result = fit(
    "y ~ x + (1 | g)",
    data=df,
    family="gaussian",
    priors={
        "intercept": "Normal(0.0, 10.0)",
        "beta_x": "Normal(0.0, 1.0)",
        "sigma_g_Intercept": "HalfCauchy(0.5)",
    },
)
```

A prior's key is the latent's name in the generated QVR source, which `formula_to_qvr`
shows before fitting.

## Diagnostics: `quivers.diagnostics`

Chain diagnostics such as $\hat R$ and the effective sample size assess how well
[MCMC](inference-mcmc.md#mcmc-hmc-and-nuts) explored the posterior.
[Posterior predictive](inference-svi.md#predictive-sampling) checks ask whether
simulated data reproduce chosen features of the observations [@stanPredictiveChecks],
and predictive comparison estimates performance on held-out observations. Chains can
agree closely on a normal model's parameters while its replicated data miss the
extreme values of the observations.

Leave-one-out (LOO) comparison estimates how well the model predicts each observation
when that observation is left out of the fit, and Pareto-smoothed importance sampling
(PSIS) approximates it by reweighting posterior draws [@vehtari2017loo]. A large
estimated Pareto shape marks an observation whose reweighting is unstable, for which
refitting without it gives a more reliable estimate. Stacking chooses model weights
that maximize the leave-one-out predictive performance of the combination
[@yao2018stacking].

A posterior predictive p-value is the probability that a statistic of replicated data
exceeds the same statistic of the observed data [@gelman1996predictive]. A mean check
asks whether replicated means tend to fall above or below the observed mean, and a value
near one half can coexist with a poor fit to the variance, so the statistics should be
the features the model is meant to explain.

The [diagnostics adapter](../api/diagnostics/index.md) converts quivers' inference
records for [ArviZ](https://python.arviz.org/) and returns an
[`xarray.DataTree`](https://docs.xarray.dev/en/stable/generated/xarray.DataTree.html)
with ArviZ-compatible groups.

<!-- python: skip -->
```python
import arviz as az
from quivers.diagnostics import (
    to_datatree, compare, loo, posterior_predictive_check,
)

idata_a = to_datatree(
    fit_a.posterior,
    observed_data={"y": y_obs},
    posterior_predictive={"y": pp_a},
    log_likelihood={"y": ll_a},
    coords={"Verb": ["eat", "drink", "run"]},
    dims={"beta": ["Verb"]},
)
idata_b = to_datatree(fit_b.posterior, ...)

# Pointwise PSIS-LOO on ArviZ's canonical log scale.
loo_result = loo(idata_a, pointwise=True, var_name="y")
print(loo_result.elpd_i, loo_result.pareto_k)

# Ranked comparison + full-precision stacking weights.
print(compare(
    {"a": idata_a, "b": idata_b},
    var_name="y",
    round_to="none",
))

# Posterior-predictive p-value on a chosen test statistic.
result = posterior_predictive_check(
    idata_a, observed_name="y", statistic="mean", by="Verb"
)
print(result["ppp"])

# Forest plot, trace plot, energy plot, etc. all consume the DataTree.
az.plot_forest(idata_a, var_names=["beta_x"])
```

[`to_datatree`](../api/diagnostics/arviz_io.md#quivers.diagnostics.arviz_io.to_datatree)
fills the ArviZ groups (`posterior`, `sample_stats`,
`posterior_predictive`, `log_likelihood`, `observed_data`,
`constant_data`) from the `(num_chains, num_samples, *site_shape)`
tensors that
[`MCMCResult`](../api/inference/mcmc.md) already produces, with its
sampler statistics of each draw under ArviZ's names (`diverging`,
`tree_depth`, `n_steps`, `energy`, `step_size`, `acceptance_rate`).
[`compare`](../api/diagnostics/comparison.md#quivers.diagnostics.comparison.compare)
delegates to
[`arviz.compare`](https://python.arviz.org/en/stable/api/generated/arviz.compare.html)
with stacking weights. Its `round_to` parameter is ArviZ's rounding policy:
`"auto"` is the default for display, an integer selects decimal
places, and `"none"` preserves full numeric precision. The
[`loo` wrapper](../api/diagnostics/comparison.md#quivers.diagnostics.comparison.loo)
delegates PSIS to [`arviz.loo`](https://python.arviz.org/en/stable/api/generated/arviz.loo.html);
with `pointwise=True`, its `elpd_i` and `pareto_k` arrays contain
one value per observation without filtering nonfinite Pareto estimates.
[`posterior_predictive_check`](../api/diagnostics/predictive_checks.md#quivers.diagnostics.predictive_checks.posterior_predictive_check)
computes the posterior predictive p-value for a chosen
[test statistic](../api/diagnostics/predictive_checks.md),
optionally grouped by a named dim.

ArviZ computes LOO, comparison, and the predictive checks.

### Convergence diagnostics for any draws

[`rhat`](../api/diagnostics/convergence.md#quivers.diagnostics.convergence.rhat),
[`ess`](../api/diagnostics/convergence.md#quivers.diagnostics.convergence.ess),
[`mcse`](../api/diagnostics/convergence.md#quivers.diagnostics.convergence.mcse), and
[`summary`](../api/diagnostics/convergence.md#quivers.diagnostics.convergence.summary)
take any tensor laid out `(num_chains, num_draws, *event_shape)` and
return one value per event coordinate. They follow the definitions of
Vehtari et al. [@vehtari2021rhat] and need only torch, which is why
[`MCMCResult`](../api/inference/mcmc.md) computes its `r_hat`, `ess`,
and `ess_tail` through them. A quantity derived from posterior draws (a
ratio of parameters, a predictive probability, a label-invariant
summary of a mixture) and the output of a hand-written sampling loop
are thus diagnosed with the estimators a run reports, and the numbers
are comparable.

Each estimator answers a different question.
[R-hat](https://mc-stan.org/docs/reference-manual/analysis.html#potential-scale-reduction)
compares split halves of the chains after replacing every draw by the
normal score of its rank, and reports the larger of that statistic and
the same statistic for the absolute deviations from the median. The
second detects chains that agree on location but disagree on scale.
Values near 1 are consistent with mixing, and Vehtari et al. suggest 1.01 as a
threshold. Bulk
[effective sample size](https://mc-stan.org/docs/reference-manual/analysis.html#effective-sample-size.section)
measures precision for location summaries such as the mean and the
median, while tail ESS (the smaller of the ESS values for the 5% and
95% quantiles) measures precision for interval endpoints, which a
sampler that explores the tails slowly may estimate far less precisely
than the bulk. The Monte Carlo standard error converts an ESS into the
error of one summary: `mcse(draws, method="mean")`, `method="sd"`, or
`method="quantile", prob=p`.

```python
import torch
from quivers.diagnostics import ess, mcse, rhat, summary

# Four chains of a hand-written random-walk Metropolis sampler for a
# standard normal target, started from dispersed points.
generator = torch.Generator().manual_seed(0)
num_chains, num_draws = 4, 2000
position = 3.0 * torch.randn(num_chains, generator=generator)
draws = torch.empty(num_chains, num_draws)
for t in range(num_draws):
    proposal = position + 0.8 * torch.randn(num_chains, generator=generator)
    log_ratio = 0.5 * (position**2 - proposal**2)
    accept = torch.rand(num_chains, generator=generator).log() < log_ratio
    position = torch.where(accept, proposal, position)
    draws[:, t] = position

print(rhat(draws))                              # close to 1
print(ess(draws), ess(draws, method="tail"))    # far below the 8000 draws
print(mcse(draws, method="quantile", prob=0.95))

# A derived quantity keeps the chain and draw axes.
second_moment = draws**2
print(rhat(second_moment), ess(second_moment))

report = summary({"x": draws, "x_squared": second_moment})
print(report["x"].quantiles, report["x"].mcse_quantiles)
```

[`summary`](../api/diagnostics/convergence.md#quivers.diagnostics.convergence.summary)
returns one
[`ConvergenceSummary`](../api/diagnostics/convergence.md#quivers.diagnostics.convergence.ConvergenceSummary)
per site, holding the posterior mean, standard deviation, and
quantiles beside every diagnostic above, so `summary(result.samples)`
tabulates a whole MCMC run.

Some inputs have no meaningful diagnostic, and the estimators return
NaN for them rather than a number that would read as valid: a
coordinate with a NaN or infinite draw, fewer than four draws per
chain, or a constant value (and, for R-hat, fewer than two chains).
The constant case departs from ArviZ, which reports the total draw
count as the ESS of a constant chain, so that a sampler stuck at its initial
value would look perfectly mixed. Chains that are each
constant but sit at different values give an infinite R-hat. On every
other input the estimators agree with ArviZ's `rhat`, `ess`, and
`mcse` to floating-point rounding.

### Simulation-based calibration

Convergence diagnostics say whether chains agree with each other, not
whether they agree with the posterior. A sampler with a bug in its
acceptance step, or a guide whose family cannot represent the
posterior, can mix well around the wrong distribution.
[Simulation-based calibration](../api/diagnostics/calibration.md) (SBC)
tests the procedure itself [@talts2018sbc]: draw parameters from the
prior, draw data from the program given those parameters, fit the
data, and record the rank of each true parameter among its posterior
draws. If the procedure samples the exact posterior, each rank is
uniform on `{0, ..., L}` for `L` posterior draws, whatever the program.
Ranks piled at both ends mean posteriors that are too narrow, ranks piled in
the middle mean posteriors that are too wide, and ranks tilted to one side mean bias.

[`sbc`](../api/diagnostics/calibration.md#quivers.diagnostics.calibration.sbc)
runs the loop for a program, an input with one row, and a fit
`fit(observations, seed)` that returns an
[`MCMCResult`](../api/inference/mcmc.md), a fitted
[`Guide`](../api/inference/guide.md), or a mapping of site names to
`(num_chains, num_draws, ...)` draws from a custom sampler. The example
below checks a fit that samples the closed-form posterior of a
Normal mean, so its ranks should look uniform.

```python
import torch
from quivers.diagnostics import sbc, sbc_coverage, sbc_uniformity
from quivers.dsl import loads

program = loads("""
object Obs : FinSet 8

program normal_mean : Obs -> Obs
    sample mu <- Normal(loc=0.0, scale=1.0)
    observe y : Obs <- Normal(loc=mu, scale=0.5)
    return y

export normal_mean
""").morphism

def exact_fit(observations, seed):
    # Conjugate posterior of mu: precision 1 + 8 / 0.25.
    precision = 1.0 + 8 / 0.25
    mean = observations["y"].sum() / 0.25 / precision
    generator = torch.Generator().manual_seed(seed)
    draws = mean + torch.randn(1, 99, generator=generator) / precision**0.5
    return {"mu": draws}

result = sbc(
    program, torch.zeros(1, 1), exact_fit,
    observed_names={"y"}, num_simulations=200, thin=1, seed=0,
)
print(result.statistic_names, result.ranks.shape, result.num_draws)
print(sbc_uniformity(result.ranks, result.num_draws))   # p-values, one per statistic
coverage = sbc_coverage(result.ranks, result.num_draws, mass=0.9)
print(coverage.null_mass, coverage.coverage, coverage.consistent)
```

With MCMC the fit passes its seed to the driver, and `thin="ess"`
(the default) keeps about one draw per effective sample, since
autocorrelated draws make even an exact sampler's ranks non-uniform:

<!-- python: skip -->
```python
from quivers.inference import MCMC, NUTSKernel

def nuts_fit(observations, seed):
    driver = MCMC(NUTSKernel(), num_warmup=500, num_samples=1000, seed=seed)
    return driver.run(program, torch.zeros(1, 1), observations)

result = sbc(program, torch.zeros(1, 1), nuts_fit,
             observed_names={"y"}, num_simulations=200, seed=0)
```

Three details decide whether the check has the size it claims.
First,
[`sbc_ranks`](../api/diagnostics/calibration.md#quivers.diagnostics.calibration.sbc_ranks)
breaks ties between a true value and its posterior draws uniformly at
random. Without that, a discrete quantity (a count, an indicator) has
non-uniform ranks even under an exact sampler. Second,
[`sbc_uniformity`](../api/diagnostics/calibration.md#quivers.diagnostics.calibration.sbc_uniformity)
compares the ranks with the discrete uniform distribution on
`{0, ..., L}`: its statistic is the smallest two-sided binomial tail
probability of the rank
[ECDF](https://en.wikipedia.org/wiki/Empirical_distribution_function)
over every threshold, the quantity behind the simultaneous bands of
Säilynoja et al. [@sailynoja2022graphical], and its p-value comes from
ranks simulated exactly under that null, where a continuous
Kolmogorov-Smirnov test would have the wrong size at small `L`. Each
statistic gets its own p-value, so compare the smallest with
`alpha / num_statistics`. Third,
[`sbc_coverage`](../api/diagnostics/calibration.md#quivers.diagnostics.calibration.sbc_coverage)
counts how often the true value falls in the central rank band, whose
exact null mass (`null_mass`) differs from the requested `mass` at
finite `L`, and gives each statistic a Clopper-Pearson interval
[@clopper1934binomial] with a Bonferroni adjustment across
statistics.

By default every coordinate of every latent site is ranked. The
`statistics=` argument names other test quantities, each a function of
a batch of parameter draws and the data, and quantities that read the
data, such as a residual sum of squares or a log-likelihood, can
detect miscalibration that the parameters alone miss
[@modrak2025sbc].
[`to_datatree_from_sbc`](../api/diagnostics/arviz_io.md#quivers.diagnostics.arviz_io.to_datatree_from_sbc)
writes the ranks into the `prior_sbc` group that ArviZ's
`plot_ecdf_pit` reads, for the graphical version of the uniformity
test.

## Initialization tools {#algebra-guided-training-tooling}

`quivers.analysis` reads a parsed QVR module and returns data about it: the shape of its
chain of steps, an initialization recipe for each `Algebra`, and warnings about
initializations. None of these functions changes the program.

### ChainShape

`ChainShape.from_module(module)` reads a `Module` and returns one `StepShape` record
for every program step (`latent`, `observe`, `marginalize`, or `let`), with:

- the bound variable name (`name`),
- the step `kind`,
- the source position (`source_line`, `source_col` from the
  tree-sitter parse),
- the chain `depth`: the position of the step among the stochastic steps (`latent`,
  `observe`, `marginalize`), counted from 1. A `let` has the depth of the most recent
  stochastic step before it, or `0` if there is none,
- the governing `algebra_name` (read from the surrounding
  `composition <name> [level=<algebra>]` declaration),
- the inferred `intermediate_size` after the step's [contraction](dsl-contractions.md#type-driven-wiring-inference)
  / activation.

<!-- python: skip -->
```python
from quivers.dsl import parse
from quivers.analysis import ChainShape

module = parse(source)
shape = ChainShape.from_module(module)

for step in shape.steps:
    print(f"{step.name} @ {step.source_line}:{step.source_col} "
          f"depth={step.depth} dim={step.intermediate_size}")
```

`recommend_init` and `saturation_warnings` below read these records.

### Algebra.init_spec

[`Algebra.init_spec(depth, intermediate_size) -> InitSpec`](../api/core/algebras.md)
gives an initialization recipe for each `Algebra`. Noisy-OR over $k$ independent terms
of probability $p$ is $1 - (1 - p)^k \approx 1 - e^{-kp}$, which is $1/2$ at
$p = \ln(2)/k$, so under `ProductFuzzy` the recipe starts entries near
$\ln(2)/k$ with $k$ = `depth * intermediate_size`. Boolean and Gödel weights start at
$1/2$, probability kernels at $1/k$ for $k$ outputs, and `REAL` uses the scale of He
et al. for rectifier networks [@he2015delving].

<!-- python: skip -->
```python
from quivers.core.algebras import ProductFuzzyAlgebra
# Importing quivers.analysis attaches init_spec to Algebra.
import quivers.analysis  # noqa: F401

algebra = ProductFuzzyAlgebra()
spec = algebra.init_spec(depth=6, intermediate_size=1)
# InitSpec(distribution='uniform', mean=0.1155, std=0.0578, ...)
# noisy-OR over k=6 cells lands at 1/2 when each cell is p ≈ ln(2)/6.
```

The option `[init=auto]` on a learnable table applies this recipe at compile time
([DSL declarations](dsl-declarations.md#init-recipes-initauto)).

### recommend_init and apply_init_spec

`recommend_init(module)` gives an `InitSpec` for each latent by calling the `Algebra`'s
`init_spec` with the step's `ChainShape` depth and intermediate size. The depth is the
step's position among the stochastic steps, so the recipe's $k$ is that position times
the size, not a count of noisy-OR terms in the model. `apply_init_spec(tensor, spec)`
writes a spec into one learnable tensor, through
[`torch.nn.init`](https://docs.pytorch.org/docs/stable/nn.init.html) for the standard
distributions.

<!-- python: skip -->
```python
from quivers.dsl import parse, loads
from quivers.analysis import recommend_init, apply_init_spec

module = parse(source)
program = loads(source)

specs = recommend_init(module)
for name, param in program.named_parameters():
    if name in specs:
        apply_init_spec(param, specs[name])
```

This is what `[init=auto]` does at compile time, and calling `recommend_init` afterward
applies the recipes to a program compiled with the default random initialization.

### saturation_warnings

A transformation saturates when it becomes insensitive to its input: a sigmoid far into
either tail has a small derivative, a long product drives scores toward zero, and
repeated noisy-OR drives them toward one. The initialization recipes aim to start in a
range with usable gradients.

`saturation_warnings(module)` returns a warning, located at its source line and column,
for each latent whose recommended `InitSpec` differs from the default `Normal(0, 1)`
initialization. It considers every `latent` step at depth at least 2 or with
intermediate size above 1, and warns when the recipe's mean differs from $0$ by more
than $0.2$ or its standard deviation from $1$ by more than $20\%$, the same cutoffs for
every `Algebra`. A warning thus says that the default initialization is far from the
recipe. It does not measure saturation during training, which shows in the scores and
their gradients, clustered near zero or one with tiny gradients.

<!-- python: skip -->
```python
from quivers.dsl import parse
from quivers.analysis import saturation_warnings

module = parse(source)
for warning in saturation_warnings(module):
    print(f"{warning.name} @ {warning.source_line}:{warning.source_col} "
          f"{warning.message()}")
```



## From a module back to source: `quivers.dsl.emit` {#module-to-source-quiversdslemit}

[`module_to_source`](../api/dsl/emit.md) writes a [`Module`](../api/dsl/ast_nodes.md)
AST as `.qvr` source, covering every declaration, program step, and expression form.
Compiling the result with [`quivers.dsl.loads`](../api/dsl/parser.md) gives a program
with the same declarations as the AST.

## Autograd-safe morphism transforms

Operations that derive a morphism from another (`change_base`,
[`.dagger`](../api/core/morphisms.md#quivers.core.morphisms.Morphism.dagger),
[`.trace`](../api/core/morphisms.md#quivers.core.morphisms.Morphism.trace),
[`.refactor`](../api/core/morphisms.md#quivers.core.morphisms.Morphism.refactor))
return a
[`TransformedMorphism`](../api/core/morphisms.md#quivers.core.morphisms.TransformedMorphism)
whose `.tensor` is recomputed from the source's tensor on each
access, and whose `.module()` registers the source as a submodule
so `.parameters()` walks reach the upstream learnable parameters.
Each access of `.tensor` builds a new autograd graph, so gradients reach the source's
parameters on every step of an optimization.
[`ObservedMorphism`](../api/core/morphisms.md#quivers.core.morphisms.ObservedMorphism),
by contrast, holds a fixed tensor of data.

### Multi-step optimization pattern

Keeping the source in the module tree keeps its parameters registered, but the
derivatives still depend on the transformation: a hard threshold makes discrete
decisions and passes no useful gradient through them. The example below trains a
learnable table `W` through `W.change_base(...).dagger` with an identity map, so the
gradient reaches `W`'s parameter on every step.

<!-- python: skip -->
```python
import torch
from quivers.dsl import loads
from quivers.core.algebra_morphisms import IdentityHom

program = loads('''
composition product_fuzzy [level=algebra]
object X : FinSet 8
object Y : FinSet 8
morphism W : X -> Y [role=latent, init=auto]
export W
''')
W = program.morphism
phi = IdentityHom(W.algebra)
target = torch.eye(8)

opt = torch.optim.Adam(program.parameters(), lr=1e-2)
for step in range(100):
    opt.zero_grad()
    # Recomputed each access: fresh autograd graph.
    rebased = W.change_base(phi)
    loss = (rebased.dagger.tensor - target).pow(2).sum()
    loss.backward()
    opt.step()
```

## End-to-end example

<!-- python: skip -->
```python
import pandas as pd
from quivers.formulas import fit
from quivers.diagnostics import (
    to_datatree, compare, posterior_predictive_check,
)
from quivers.analysis import saturation_warnings

df = pd.read_csv("acceptability.csv")

# Fit a hierarchical logistic model with a polynomial predictor.
fit_full = fit(
    "response ~ poly(rt, 2) + verb + (1 + verb | subject)",
    data=df,
    family="bernoulli",
    method="nuts",
    num_warmup=500, num_samples=1000, num_chains=4,
    priors={"sigma_subject_Intercept": "HalfCauchy(0.5)"},
    seed=0,
)

# Check init saturation on the emitted QVR source.
from quivers.dsl import parse
for warning in saturation_warnings(parse(fit_full.qvr_source)):
    print(warning)

# Compare against a simpler null model.
fit_null = fit(
    "response ~ verb + (1 | subject)",
    data=df, family="bernoulli", method="nuts",
    num_warmup=500, num_samples=1000, num_chains=4, seed=0,
)

idata_full = to_datatree(fit_full.posterior, ...)
idata_null = to_datatree(fit_null.posterior, ...)
print(compare({"full": idata_full, "null": idata_null}))

# Per-verb posterior-predictive p-value (PPP) on the response rate.
print(posterior_predictive_check(
    idata_full, observed_name="response", statistic="mean", by="verb",
))

# Inspect / save the emitted QVR.
fit_full.dump_qvr("acceptability_model.qvr")
```

When a formula cannot express a model, `formula_to_qvr(...)` emits the program, the
source can be edited, and [`quivers.dsl.loads`](../api/dsl/parser.md) compiles the
edited version.

## See also

- [Data and Formulas](analysis-data-and-formulas.md): data schemas and formulas.
- [Variational Inference: SVI](inference-svi.md) and
  [MCMC](inference-mcmc.md): the inference drivers `fit(...)`
  wraps.
- [DSL Declarations](dsl-declarations.md#init-recipes-initauto): the `[init=auto]`
  option, which applies `Algebra.init_spec` at compile time.
