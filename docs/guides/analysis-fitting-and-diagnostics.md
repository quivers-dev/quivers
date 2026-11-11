# Analysis: Fitting and Diagnostics

After specifying a model, use the
`fit(...)` entry point to estimate its posterior. The sections below cover
[ArviZ](https://python.arviz.org/) diagnostics, initialization for each
algebra, and morphism transforms that preserve gradients. [Data and Formulas](analysis-data-and-formulas.md)
covers data loading and the formula DSL.

## One-line fit

A fit combines three choices: a model (formula, family, and priors),
an inference method, and data. The returned object preserves the model
source as well as the approximation to its posterior. Inspecting the
source shows which parameters were introduced and which data columns
they use. The diagnostics below assess chain exploration and compare
replicated observations with the supplied data.

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

The returned [`BayesianFit`](../api/formulas/fit.md) is itself a
frozen [`dx.Model`](https://didactic.dev/api/Model), with
`.formula`, `.family`, `.program` (the compiled
    [`MonadicProgram`](../api/continuous/programs.md)), `.posterior`
    (either an [`MCMCResult`](../api/inference/mcmc.md) or a
[`Guide`](../api/inference/guide.md)), and `.observations` (the
inference-time observations dict).

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

Prior keys reference the latent name as it appears in the emitted
QVR source; use `formula_to_qvr` to inspect names upfront.

## Diagnostics: `quivers.diagnostics`

Chain diagnostics such as
R-hat and effective sample size assess aspects of [MCMC exploration](inference-mcmc.md#mcmc-hmc-and-nuts).
[Posterior predictive](inference-svi.md#predictive-sampling) checks ask whether simulated data reproduce chosen
[features of the observations](https://mc-stan.org/docs/stan-users-guide/posterior-predictive-checks.html#which-statistics-to-test) [@stanPredictiveChecks]. Predictive comparison estimates performance
on held-out observations. For instance, chains may agree closely on a
Normal model’s parameters while its replicated data miss the extreme
values in the observations.

**Leave-one-out (LOO)** comparison estimates prediction for each observation
when it is left out of the fit. **Pareto-smoothed importance sampling
(PSIS)** approximates that calculation by reweighting posterior draws.
Large Pareto shape estimates indicate unstable reweighting. For those
observations, refitting with the observation omitted can give a more
reliable predictive comparison. Stacking chooses weights to improve the
combined predictive distribution on held-out data.

A posterior predictive p-value compares a statistic of replicated data
with the same statistic of observed data. For instance, a mean-response
check asks whether replicated means tend to fall above or below the
observed mean. A value near one half can coexist with a poor fit to the
variance: two distributions may have the same mean and very different
spreads. Choose statistics that reflect the features your model should explain.

The [diagnostics adapter](../api/diagnostics/index.md) connects
between quivers' inference records and
[ArviZ](https://python.arviz.org/). It returns an
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
populates the canonical ArviZ groups (`posterior`, `sample_stats`,
`posterior_predictive`, `log_likelihood`, `observed_data`,
`constant_data`) from the `(num_chains, num_samples, *site_shape)`
tensors that
[`MCMCResult`](../api/inference/mcmc.md) already produces, with its
per-draw sampler statistics under ArviZ's names (`diverging`,
`tree_depth`, `n_steps`, `energy`, `step_size`, `acceptance_rate`).
[`compare`](../api/diagnostics/comparison.md#quivers.diagnostics.comparison.compare)
delegates to
[`arviz.compare`](https://python.arviz.org/en/stable/api/generated/arviz.compare.html)
with stacking weights
[@yao2018stacking].
Its `round_to` parameter is ArviZ's rounding policy:
`"auto"` is the display-oriented default, an integer selects decimal
places, and `"none"` preserves full numeric precision. The
[`loo` wrapper](../api/diagnostics/comparison.md#quivers.diagnostics.comparison.loo)
delegates PSIS to [`arviz.loo`](https://python.arviz.org/en/stable/api/generated/arviz.loo.html);
with `pointwise=True`, its `elpd_i` and `pareto_k` arrays contain
one value per observation without filtering nonfinite Pareto estimates.
[`posterior_predictive_check`](../api/diagnostics/predictive_checks.md#quivers.diagnostics.predictive_checks.posterior_predictive_check)
computes the canonical
[posterior-predictive p-value](https://en.wikipedia.org/wiki/Posterior_predictive_p-value)
for a user-chosen
[test statistic](../api/diagnostics/predictive_checks.md),
optionally grouped by a named dim.

No information-criterion math is reimplemented here; LOO, comparison,
and predictive checks come from ArviZ.

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
the same statistic for the absolute deviations from the median; the
second detects chains that agree on location but disagree on scale.
Values near 1 are consistent with mixing, and Vehtari et al. suggest
1.01 as a threshold. Bulk
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
count as the ESS of a constant chain; a sampler stuck at its initial
value would otherwise look perfectly mixed. Chains that are each
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
Ranks piled at both ends mean posteriors that are too narrow; piled in
the middle, too wide; tilted to one side, biased.

[`sbc`](../api/diagnostics/calibration.md#quivers.diagnostics.calibration.sbc)
runs the loop for a program, a one-row input, and a fit
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
random; without that, a discrete quantity (a count, an indicator) has
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
a batch of parameter draws and the data; quantities that read the
data, such as a residual sum of squares or a log-likelihood, can
detect miscalibration that the parameters alone miss
[@modrak2025sbc].
[`to_datatree_from_sbc`](../api/diagnostics/arviz_io.md#quivers.diagnostics.arviz_io.to_datatree_from_sbc)
writes the ranks into the `prior_sbc` group that ArviZ's
`plot_ecdf_pit` reads, for the graphical version of the uniformity
test.

## Algebra-guided training tooling

The `quivers.analysis` subpackage
collects static-analysis passes that read a compiled QVR program
and return structured data the user can act on: chain shapes,
algebra-specific init recipes, and saturation warnings. None of
the passes here rewrite the program; they only derive metadata,
diagnostics, or sampler / init parameters that respect the source
as the canonical specification.

### ChainShape

`ChainShape.from_module(module)` walks
a compiled `Module`, returning per-step `StepShape` records that
tag every program step (`latent`, `observe`, `marginalize`, or
`let`) with:

- the bound variable name (`name`),
- the step `kind`,
- the source position (`source_line`, `source_col` from the
  tree-sitter parse),
- the chain `depth`: the 1-indexed position of the step among
  stochastic binds (`latent` / `observe` / `marginalize`). `let`
  steps inherit the depth of the most recent stochastic
  predecessor; a `let` before any stochastic step has depth `0`,
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

This metadata supplies init-recipe selection, saturation diagnostics, and
layer-width checks through `ChainShape`.

### Algebra.init_spec

Each algebra exposes an algebra-specific init recipe via
[`Algebra.init_spec(depth, intermediate_size) -> InitSpec`](../api/core/algebras.md).
The recipes target the algebra's neutral element: a `k`-way noisy-OR
under `ProductFuzzy` starts near $p \approx \ln(2) / k$ so
the aggregate stays near $1/2$; Boolean / Gödel weights start
at $1/2$; probabilistic kernels start at $1 / k$ for a $k$-way
output; Real algebras use the conventional Kaiming-style scale.

<!-- python: skip -->
```python
from quivers.core.algebras import ProductFuzzyAlgebra
# Importing the analysis package monkey-patches Algebra.init_spec on.
import quivers.analysis  # noqa: F401

algebra = ProductFuzzyAlgebra()
spec = algebra.init_spec(depth=6, intermediate_size=1)
# InitSpec(distribution='uniform', mean=0.1155, std=0.0578, ...)
# noisy-OR over k=6 cells lands at 1/2 when each cell is p ≈ ln(2)/6.
```

The `[init=auto]` annotation on a latent declaration consumes this
spec at compile time; see the
[DSL declarations guide](dsl-declarations.md#init-recipes-initauto)
for the corresponding syntax.

### recommend_init and apply_init_spec

Given a program,
`recommend_init(module)` produces a
per-latent `InitSpec` by composing `ChainShape` with each
algebra's `init_spec`.
`apply_init_spec(tensor, spec)`
materializes the spec onto a single learnable tensor, with
[`torch.nn.init`](https://docs.pytorch.org/docs/stable/nn.init.html)
integration for the standard distributions.

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

The full pipeline is what `[init=auto]` does at compile time;
calling `recommend_init` post-hoc lets you swap in algebra-guided
init on a program that was originally compiled with the default
random init.

### saturation_warnings

Saturation occurs when a numerical transformation becomes insensitive
to its input: a sigmoid far into either tail has a small derivative.
A long product can likewise drive scores toward zero, while repeated
noisy-OR can drive them toward one. Initialization recipes aim to keep
early calculations in a range with usable gradients. After a warning,
inspect the affected scores and their gradients during training; scores
clustered near zero or one with tiny gradients can motivate a different
initialization or parameterization.

`saturation_warnings(module)` returns
source-keyed warnings about latents whose algebra-guided init
recipe differs materially from the default `Normal(0, 1)` init. It
walks every `latent` step at depth $\geq 2$ or with intermediate
size $> 1$, asks the shape module for the recommended `InitSpec`,
and flags the step when either the recipe's mean shifts by more
than $0.2$ from $0$ or its std moves by more than $20\%$ away from
$1$. The 20% threshold is a single generic cutoff, applied
identically across ProductFuzzyAlgebra, BooleanAlgebra, Godel,
Markov, and other supported algebras.

<!-- python: skip -->
```python
from quivers.dsl import parse
from quivers.analysis import saturation_warnings

module = parse(source)
for warning in saturation_warnings(module):
    print(f"{warning.name} @ {warning.source_line}:{warning.source_col} "
          f"{warning.message()}")
```

Use these warnings as a static-analysis pass during model
development: a model that fits but trains slowly often turns out
to be saturated under its default init.

## Module-to-source: `quivers.dsl.emit`

[`module_to_source`](../api/dsl/emit.md) walks a
[`Module`](../api/dsl/ast_nodes.md) AST and produces canonical
`.qvr` source for every current statement, program-step, and
expression variant. The emit is one-way and semantic: passing the
emitted source to [`quivers.dsl.loads`](../api/dsl/parser.md) compiles a
program with the same declarations as the original AST.

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
Every backward through a fresh `.tensor` access gets its own
autograd graph, so multi-step optimization propagates gradients
through the V-Cat representation correctly.

[`ObservedMorphism`](../api/core/morphisms.md#quivers.core.morphisms.ObservedMorphism)
is the wrapper for fixed data tensors (the
`from_data(...)` path); it does not double as the wrapper for
derived-from-source morphisms. The categorical distinction "this
morphism is a tensor function of another morphism" earns the
separate
[`TransformedMorphism`](../api/core/morphisms.md#quivers.core.morphisms.TransformedMorphism)
class.

### Multi-step optimization pattern

Keeping the source in the module tree preserves parameter registration,
but derivatives still depend on the transformation itself. A hard
threshold returns discrete decisions and has no useful gradient through
those decisions. The example uses a differentiable conversion; train
with an appropriate smooth objective before applying a hard threshold
for reporting or a later discrete calculation.

The pattern below trains a latent `W` and uses
`W.change_base(...).dagger` with a differentiable identity conversion in the loss; both passes through the
optimizer see the gradient flowing back through `W`'s underlying
parameter.

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

The same workflow drops down to the QVR DSL whenever the formula
language is too restrictive: `formula_to_qvr(...)` emits the
program, the user edits the source, and feeds it back through
[`quivers.dsl.loads`](../api/dsl/parser.md).

## See also

- [Data and Formulas](analysis-data-and-formulas.md): the front-half
  of the analysis stack.
- [Variational Inference: SVI](inference-svi.md) and
  [MCMC](inference-mcmc.md): the inference drivers `fit(...)`
  wraps.
- [DSL Declarations](dsl-declarations.md#init-recipes-initauto):
  the `[init=auto]` annotation that consumes
  `Algebra.init_spec` at compile time.

## Further reading

- [@yao2018stacking]
