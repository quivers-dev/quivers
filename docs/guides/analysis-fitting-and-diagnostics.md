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
[`dx.Model`](https://didactic.dev/api/Model) with `.formula`, `.family` (the family
given), `.residual` (the residual dependence of an autocorrelation term such as
`ar()`, or `None` for independent rows), `.program`
(the compiled [`MonadicProgram`](../api/continuous/programs.md)), `.posterior` (an
[`MCMCResult`](../api/inference/mcmc.md) or a [`Guide`](../api/inference/guide.md)),
and `.observations` (the observations dict used for inference).

The `seed` argument to [`fit`](../api/formulas/fit.md) controls both model
initialization and inference. Compilation and fitting run in an isolated CPU
random state, and the caller's state is restored even if fitting raises an error.
For MCMC, the same argument initializes the driver's independent chain streams.

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

An existing mapping override uses the latent name shown by `formula_to_qvr`.
Typed `PriorSpec`/`PriorSet` rules also select parameter classes, coefficients,
groups, distributional parameters, and nonlinear leaves through retained
metadata. The [selector workflow](analysis-data-and-formulas.md#prior-selectors-and-shared-shrinkage)
covers precedence, normalized bounds, and shared shrinkage without requiring
generated site names.

### Prior-only fitting and independent prior draws

Both formula `fit` and source `Model.fit` accept `sample_prior="no"`, `"yes"`,
or `"only"`. The default fits the posterior. `"yes"` fits the same posterior
and retains an independent prior stream; `"only"` runs the selected inference
method with observation likelihoods disabled. Data-dependent priors, support
constraints, and authored score potentials remain active. Compiler-generated
marginal likelihood factors follow the observation policy as well.

```python
import pandas as pd
from quivers.formulas import fit

training = pd.DataFrame({"x": [-1.0, 0.0, 1.0], "y": [0.2, -0.1, 0.8]})
result = fit(
    "y ~ x", data=training, method="svi", num_steps=2, seed=23,
    sample_prior="yes", num_prior_samples=4, prior_seed=19,
)
prior = result.prior_view()
expected = prior.posterior_epred()
replicated = prior.posterior_predict(seed=18)
assert expected.shape == replicated.shape == (1, 4, 3)
```

`prior_view()` reuses retained samples; `prior_view(num_samples=100, seed=7)`
draws a fresh independent stream. Source fits expose the same view, and a bound
source model or formula fit also offers `prior_draws`. Priors are drawn through
the original conditional laws, rather than by reinitializing a fitted guide.
Separate prior seeds preserve posterior draws, optimization losses, and the
caller's CPU random state. Independent prior samples carry no MCMC diagnostics.

An authored score potential can couple otherwise ancestral draws. Independent
sampling refuses that case instead of treating weighted draws as independent
samples from a normalized law. Use `prior_draws(method=MCMC(...))` or
`prior_view(method=MCMC(...))` with an explicitly configured driver for
factor-conditioned prior-only inference. Raw recorded-event likelihoods remain
available through `log_likelihood`, including from prior views; weights and the
fitting score policy do not change their definition. See
[prior sampling](../api/inference/prior.md) and the
[observation scoring policy](../api/effects/observation_scoring.md).

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

Both source and formula fits expose `analysis(num_samples=100, seed=7)`.
The returned object retains posterior draws and raw response likelihoods for
`summary`, `waic`, `loo`, and `to_datatree`, without replaying the model for
each operation. Recorded MCMC chains supply convergence diagnostics;
variational draws supply uncertainty summaries without convergence claims.
The rows of a `marginalize` block have scores of their own only when no two of
them share a latent class. Otherwise the source fit's `analysis` raises
[`UnfactoredObservationError`](../api/effects/units.md#quivers.effects.units.UnfactoredObservationError),
and `analysis(units="factors")` retains one score per independent unit instead,
the scores that
[`ModelFit.unit_log_likelihood`](../api/modeling.md#quivers.modeling.model.ModelFit.unit_log_likelihood)
returns, so `loo` and `waic` leave out one unit at a time
([units and row scores](../semantics/programs.md#271-units-and-row-scores)).
The [executable workflow](analysis-data-and-formulas.md#raw-likelihoods-and-retained-analysis)
shows the shared interface. Lower-level samplers can use the adapters below
when they need custom coordinates or predictive groups.

Joint parameter dependence uses the same retained analysis. `covariance`
returns covariance and correlation together; it does not draw again. Site
names and event indices remain separate, including when labels contain
punctuation:

```python
import torch
from quivers.diagnostics import PosteriorAnalysis

retained = PosteriorAnalysis(
    samples={
        "intercept": torch.tensor([[0.1, 0.3, 0.2, 0.5]]),
        "slope": torch.tensor([[[0.2, 0.5], [0.4, 0.6], [0.1, 0.3], [0.6, 0.8]]]),
    },
    log_likelihood={}, kind="variational",
)
dependence = retained.covariance(
    dims={"slope": ["condition"]},
    coords={"condition": ["control", "treatment"]},
)
assert dependence.covariance.shape == dependence.correlation.shape == (3, 3)
assert dependence.coordinates == (("intercept", ()), ("slope", (0,)), ("slope", (1,)))
assert dependence.coordinate_labels == ((), ("control",), ("treatment",))
```

For a fitted model, replace `retained` with either source or formula
`result.analysis(...)`. The default pools posterior chains and draws;
`chainwise=True` retains one matrix per chain. `ddof=1` uses sample covariance.
Missing values fail explicitly, and constant coordinates have NaN correlations.
These matrices describe dependence among draws; they do not establish MCMC
convergence. The [API reference](../api/diagnostics/posterior.md#joint-covariance-and-correlation)
states the coordinate and numeric-capacity policies.

A derived contrast can use those same retained tensors. Pass its
`(chain, draw, *event)` tensor to `hypothesis_summary`, optionally with jointly
constructed prior contrasts. The result separates below, equal, and above
frequencies from posterior region odds and prior-corrected region Bayes factors.
Compute a contrast within each joint draw; independently rearranging parameter
draws would change their dependence. Exact equality is descriptive, without a
point-null density estimate. The
[contrast API](../api/diagnostics/hypothesis.md) states the prior assumptions,
coordinate labels, and empirical zero-frequency policy.

The [diagnostics adapter](../api/diagnostics/index.md) converts quivers' inference
records for [ArviZ](https://python.arviz.org/) and returns an
[`xarray.DataTree`](https://docs.xarray.dev/en/stable/generated/xarray.DataTree.html)
with ArviZ-compatible groups. A formula fit's `posterior_predict` method
supplies replicated responses on its retained training rows, or on a `new_data`
frame evaluated with fitted transforms and known group levels. It includes response
noise and preserves separate chain and draw axes. The
[new-row workflow](analysis-data-and-formulas.md#predictions-on-new-rows)
shows prediction without response or weight columns. The
[posterior workflow](analysis-data-and-formulas.md#posterior-coefficients-and-predictions)
distinguishes these draws from linear predictors and expected responses.

<!-- python: skip -->
```python
import arviz as az
from quivers.diagnostics import (
    to_datatree, compare, loo, posterior_predictive_check,
)

pp_a = fit_a.posterior_predict(seed=7)
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

### Score recorded fresh outcomes

`BayesianFit.log_likelihood(new_data=...)` evaluates recorded outcomes under
retained posterior draws. It reuses the fitted response transformation and
categorical level order, predictor encodings, group-effect policies, and native
observation scorer. The output preserves `(chain, draw, row)`; vector event
components are reduced within a row. Fitting weights are excluded, so fresh rows
need no weight column.

<!-- python: run -->
```python
import pandas as pd
from quivers.formulas import fit

training = pd.DataFrame({"y": [0.2, 0.9, 1.1], "x": [0.0, 1.0, 2.0]})
fitted = fit("y ~ x", data=training, method="svi", num_steps=2, seed=7)
held_out = pd.DataFrame({"y": [1.0, 1.4], "x": [2.5, 3.0]})
scores = fitted.log_likelihood(new_data=held_out, num_samples=3, seed=8)
assert scores["y"].shape == (1, 3, 2)
```

The small optimization budget illustrates the interface; inference still needs
the checks described above. Supply the response and any authored censoring,
truncation, measurement-error, decision, subset, or trial columns. Subsets select
fresh rows before response evaluation, and neural `predictor_data` follows the
same selection. Vector count responses infer fresh row totals; an explicit
`trials` declaration must agree with their component sums. Categorical outcomes
retain the training level order even when a fresh batch omits some levels.

`re_formula`, `allow_new_levels`, and `sample_new_levels` have the same meanings
as in posterior prediction. These scores condition on the selected group-effect
policy; they are not an integration over every possible new group. Omitting
`new_data` scores the fitted outcomes. Source `ModelFit.log_likelihood(data=...)`
uses its existing checked rebinding interface for the corresponding operation,
and `ModelFit.unit_log_likelihood(data=...)` scores the units of marginalized
observations whose rows share a latent.

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

Each simulation and fit uses an independently derived seed. CPU and available
accelerator random states are restored when either scope exits, including when a
fit raises an error. A callback that draws from a global Torch generator is thus
reproducible without changing the caller's random stream.

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

## brms summaries, residuals, and predictive criteria

Formula fits ([`BayesianFit`](../api/formulas/fit.md) and
[`MultivariateFit`](../api/formulas/multivariate.md)) carry brms's
post-processing methods [@burkner2017brms]. They replay the fit's retained
draws through its own prediction and scoring methods, so every table describes
the same posterior as `draws`, `posterior_predict`, and `log_likelihood`.

### Parameter summaries

`variables` returns draws under brms's parameter names: `b_<coef>` for
population coefficients (`b_sigma_<coef>` for a distributional parameter),
`b_Intercept[k]` for cumulative thresholds, the family parameter's own name
(`sigma`) when it is not regressed, `sd_<group>__<coef>` and
`cor_<group>__<a>__<b>` for group scales and correlations, and
`r_<group>[<level>,<coef>]` for group coefficients. Coefficient labels pass
through brms's renaming of special characters, so the factor column `f[q]` is
`fq` and `I(x^2)` is `IxE2`, in single and joint fits alike (`b_y_fq` for
response `y`).

Every variable is one scalar per draw, as in brms. A vector parameter is
listed per coordinate (`simo_mok1[2]`), a correlation matrix per pair
(`cor_g__Intercept__x`, `cortime__0__1`, `rescor__y__y2`), and the mixing
proportions of a mixture per component (`theta1`). The terms of a formula
share one catalogue of names:

| term | variables |
| --- | --- |
| `0 + Intercept` | `b_Intercept` |
| `s()`, `t2()` | `bs_<term>_<k>` (unpenalized columns), `sds_<term>_<j>` (penalty scales), `s_<term>_<j>[k]` (penalized coefficients), where `<term>` is the function name followed by its covariates and `by` variable (`sx` for `s(x, k = 5)`, `sxg` for `s(x, g, bs = "fs")`). Each level of a factor `by` has its own block: `sds_sxga_1` and `s_sxga_1[k]` for level `a` of `s(x, by = g)`, and `bs_sx:ga_1` for its unpenalized column |
| `gp()` | `sdgp_<term>`, `lscale_<term>`, `zgp_<term>[k]` |
| `mo()` | `bsp_mo<x>`, `simo_mo<x>1[k]` |
| `me()`, `mi()` | `bsp_me<x><sd>`, `bsp_mi<x>`, `meanme_me<x>`, `sdme_me<x>`, `Ymi_<resp>[row]` |
| `cs()` | `bcs_<x>[k]` |
| `gr(g, by = b)` | `sd_<g>__<coef>:<b><level>`, `cor_<g>__<a>:<b><level>__<c>:<b><level>` |
| `gr(g, dist = "student")` | `df_<g>`, `udf_<g>[level]` |
| `ar()`, `ma()`, `cosy()`, `unstr()` | `ar[k]`, `ma[k]`, `cosy`, `cortime__<t1>__<t2>`, with `sderr` and `err[row]` for latent residuals |
| `sar()`, `car()` | `lagsar` or `errorsar`, `car`, `sdcar`, `rhocar`, `rcar[level]` (for `type = "bym2"` the standardized components `zcar[level]` and `nszcar[level]`) |
| `mixture()` | `theta<k>`, `b_mu<k>_<coef>`, `sigma<k>` |
| `cox` | `sbhaz[k]` |
| `logistic_normal` | `sigma<category>`, `lncor__<a>__<b>` |
| horseshoe, R2D2 priors | `hs_global`, `hs_slab`, `hs_local[k]`, `R2D2_R2`, `R2D2_phi[k]`, `R2D2_tau2`, and the prior scale `sdb_<coef>` of each shrunk coefficient |

The latent values of `me()` terms are omitted, as brms omits them unless
`save_pars(latent = TRUE)` is set, and `draws` holds their innovations.

These names match brms's `variables()` for one small model per term, with
four differences.
brms's constant `disc` of an ordinal model is absent. An ordinal family other than a cumulative
one without `cs()` lists `b_Intercept` and `cutpoint_offset[k]` where brms
lists `b_Intercept[k]`. A mixture adds the positive gap
`mixture_intercept_spacing[k]` that orders its intercepts. A BYM2 term lists
its two standardized components where brms lists their weighted sum
`rcar[level]`.

The variables are listed in the order of brms's `variables()`. brms sorts
parameters by class: the population coefficients `b`, then `bs`, `bsp`,
`bcs`, the autocorrelation parameters, `sd`, `cor`, `df`, `sds`, `sdgp`,
`lscale`, the family's parameters, the shrinkage parameters, `lncor`,
`rescor`, `simo`, and last the classes with one variable per level or row
(`r`, `s`, `zgp`, `rcar`, `sbhaz`, `Ymi`, `meanme`, `sdme`). Within a class,
intercepts come first. Variables without a brms counterpart follow the
others.

Two smooths of one covariate, such as `s(x, k = 4) + s(x, bs = "cr")`, have
the same label. brms repairs the repeated names with R's `make.unique`, so the
second smooth is listed as `bs_sx_1__1`, `sds_sx_1__1`, and `s_sx_1[k]__1`,
and `variables` lists the same names.

`summary` groups the variables into brms's sections: population effects
(a smooth's unpenalized column `bs_sx_1` is the row `sx_1`), family
parameters, group-level standard deviations and correlations, smoothing
standard deviations (`sds(sx_1)` in the `splines` table), and residual
correlations of a joint fit. `fixef`, `ranef`, `coef`, `varcorr`, and `vcov`
return the corresponding brms tables, and `posterior_summary` and
`posterior_interval` summarize any named parameters.

Each table is a [`SummaryTable`](../api/diagnostics/posterior_tables.md):
labeled rows, labeled statistic columns, and, for brms's three-way arrays such
as `ranef`, a component axis. Statistics are brms's: the mean and standard
deviation, or with `robust=True` the median and R's `mad`, with type-7
quantiles. Recorded MCMC chains add `Rhat`, `Bulk_ESS`, and `Tail_ESS`
[@vehtari2021rhat]. Variational and prior draws have no convergence columns,
since a single pseudo-chain cannot diagnose mixing.

<!-- python: run -->
```python
import pandas as pd
from quivers.formulas import fit

data = pd.DataFrame({
    "y": [0.3, -1.2, 0.8, 1.9, -0.4, 0.1, 1.4, -0.7, 0.6, 2.2, -1.5, 0.9],
    "x": [0.5, -1.1, 0.3, 1.7, -0.2, 0.9, -1.4, 0.6, 1.2, -0.8, 0.4, -0.3],
    "g": ["a", "b", "c"] * 4,
})
model = fit("y ~ x + (1 | g)", data=data, method="svi", num_steps=50, seed=1)
assert list(model.fixef(num_samples=20).rows) == ["Intercept", "x"]
summary = model.summary(num_samples=20)
assert list(summary.random["g"].rows) == ["sd(Intercept)"]
assert model.ranef(num_samples=20)["g"].rows == ("a", "b", "c")
assert "residual__" in model.varcorr(num_samples=20)
```

Two brms names have no counterpart here. Native sites without a brms class,
such as the log-spacing coordinates of ordinal thresholds that are not
cumulative, keep their QVR names, one variable per coordinate. The log
posterior density, Stan's `lp__`, is
returned by `log_posterior` rather than among the variables. It is the
sampler's Jacobian-corrected log density and differs from Stan's by the
constant terms Stan drops. `nuts_params` returns the sampler statistics with
Stan's names in bayesplot's long format.

### Predictions, residuals, and draw selection

`predict`, `fitted`, and `residuals` summarize `posterior_predict`,
`posterior_epred` (or `posterior_linpred` with `scale="linear"`), and
`predictive_error`, the draws of observed minus predicted responses.
Ordinal and categorical predictions are tabulated as category frequencies, as
brms's `posterior_table` does, and their errors are refused. Pearson residuals
divide each error by the row's predictive standard deviation. `draw_ids`
selects pooled draws by 0-based position, and every prediction keyword
(`new_data`, `re_formula`, new-level policies) has its existing meaning.
brms also takes `draw_ids` on `posterior_predict`, `posterior_epred`,
`posterior_linpred`, and `log_lik`. Their counterparts here
(`log_likelihood` for `log_lik`) take `num_samples` and `seed` and have no
`draw_ids`, so a chosen set of draws is read through the summary methods
with `summary=False`, as in the last call below.

<!-- python: run -->
```python
table = model.predict(num_samples=20)
assert list(table.columns) == ["Estimate", "Est.Error", "Q2.5", "Q97.5"]
errors = model.predictive_error(method="posterior_epred", num_samples=20)
assert errors.shape == (1, 20, 12)
chosen = model.fitted(scale="linear", draw_ids=[0, 5], summary=False, num_samples=20)
assert chosen.shape == (1, 2, 12)
```

### Leave-one-out criteria

`loo` estimates the expected log pointwise predictive density (ELPD) by
Pareto-smoothed importance sampling (PSIS) through ArviZ [@vehtari2017loo;
@vehtari2024psis]. A Pareto shape above the threshold (0.7 by default, as in
brms) marks an observation whose importance weights are unreliable.
`moment_match=True` applies moment matching to those observations: affine maps
of the unconstrained draws match the moments of each leave-one-out posterior,
and the moved draws are reweighted with the fit's own log joint density
[@paananen2021implicitly]. `reloo=True` instead refits the model without each
remaining problematic observation. `waic` estimates WAIC, and `psis` returns the
smoothed weights. Each criterion is an [`ELPDEstimate`](../api/diagnostics/elpd.md)
with brms's estimates table and pointwise terms.

`loo_subsample` estimates the ELPD from a simple random subsample of rows
[@magnusson2019bayesian; @magnusson2020leave]. Every row is approximated by
its log density at the posterior mean (`"plpd"`, brms's default), only the
subsample is scored at every draw, and the difference estimator corrects the
approximation. The estimate reports the subsampling standard error separately.
The posterior mean is taken in the sampler's unconstrained coordinates, which
gives a valid point for correlation and simplex sites. brms averages
constrained draws instead. The estimator stays unbiased for either point.
Where a row's score on new data is not its training score (a residual law
that correlates rows or conditions on earlier responses, or a latent drawn for
each training row, such as latent residuals or an exact Gaussian process), the
sampled rows are scored on the training rows instead, at the cost of scoring
every row at every draw.

`loo_predict`, `loo_epred`, `loo_linpred`, and `loo_predictive_interval` weight
predictions by the same importance weights, and `loo_r2` computes brms's
leave-one-out R-squared by the Bayesian bootstrap [@gelman2019rsquared]. These
expectations assume reliable weights, and each reports a function-specific
Pareto shape.

`kfold` refits the model on each fold through the existing refit callback,
with brms's partitions: random folds, one fold per level of `group`,
`"stratified"` or `"grouped"` folds, or `"loo"` (with `group`, each fold omits
the observation's whole group). `joint` selects the predictive unit:
observations, whole folds, or groups within folds. With `save_fits=True`,
`kfold_predict` predicts each held-out row from its own refit. Folds are drawn
from a seeded quivers stream, not from R's generator, so a fold assignment
reproduces only within quivers.

`add_criterion` stores criteria on a new fit, and they persist through
`save_fit` and `load_fit`. `loo_compare`, `loo_model_weights`, and
`model_weights` compare fits by stored criteria or by precomputed estimates, and
refuse fits that score different responses.

<!-- python: run -->
```python
from quivers.formulas import loo_compare

simple = fit("y ~ 1", data=data, method="svi", num_steps=50, seed=1)
first = model.add_criterion("loo", num_samples=100)
second = simple.add_criterion("loo", num_samples=100)
table = loo_compare({"group": first, "intercept": second})
assert table.columns[:2] == ("elpd_diff", "se_diff")
assert first.criterion("loo").pareto_k.shape == (12,)
```

[`loo`](../api/formulas/index.md#quivers.formulas.loo) and
[`waic`](../api/formulas/index.md#quivers.formulas.waic) take several fits at
once, as brms's `loo(fit1, fit2)` does. Each fit's criterion is computed
unless it is stored, and the result holds the estimates and the comparison
table.

<!-- python: run -->
```python
from quivers.formulas import loo

compared = loo(model, simple, model_names=["group", "intercept"])
assert set(compared.estimates) == {"group", "intercept"}
assert compared.diffs.columns[:2] == ("elpd_diff", "se_diff")
```

A joint fit scores the units of `MultivariateFit.log_likelihood`: one row's
vector under residual correlation, or the sum over the selected responses, and
`resp` selects responses as in brms.

## Evidence, effects, and predictive checks

This section covers the brms functions that compare models by their marginal
likelihood, average models, display conditional effects, and check
predictions against data: `bridge_sampler`, `bayes_factor`, `post_prob`,
`posterior_average`, `pp_average`, `conditional_effects`, `make_conditions`,
`conditional_smooths`, `pp_check`, `mcmc_plot`, `plot`, and `pairs`
[@burkner2017brms]. Each works on single-response fits and on joint
multivariate fits (with `resp`), and each predicts through the fit's own
prediction methods.

### Marginal likelihoods by bridge sampling

The [marginal likelihood](https://en.wikipedia.org/wiki/Marginal_likelihood)
of a model is the normalizing constant

$$
p(y) = \int p(y \mid \theta)\, p(\theta)\, d\theta
$$

of its posterior. [`bridge_sampler`][quivers.formulas._evidence.bridge_sampler]
estimates it by bridge sampling [@meng1996simulating] as the R
`bridgesampling` package does when brms calls it
[@gronau2017tutorial; @gronau2020bridgesampling]. The posterior draws are
mapped to the sampler's unconstrained coordinates, where the log density is
the fit's log joint plus the log Jacobian of the constraining transforms. The
first half of every chain fits a proposal, either a normal with the draws'
mean and covariance (`method="normal"`) or the standard normal after the
third-order warp of @meng2002warp (`method="warp3"`). The second half and as
many proposal draws then enter the fixed-point iteration for the optimal
bridge function. The [`BridgeResult`][quivers.diagnostics.bridge.BridgeResult]
holds the estimate and the log densities, and `error_measures()` gives the
approximate relative mean-squared error of a `"normal"` estimate
[@fruhwirth2004estimating], or the spread of several repetitions.

Every density of a compiled formula program keeps its normalizing constant,
so the requirement brms states as `save_pars(all = TRUE)` with a normalized
Stan model holds by construction. What cannot hold is refused with an error
that names it: an improper flat prior (the marginal likelihood is not
defined), an explicit score potential whose mass is unknown, draws that are
not posterior MCMC draws (an SVI guide or prior draws), and discrete latent
sites outside the unconstrained coordinates. Proposal draws come from a
seeded Torch stream, so an estimate agrees with R's only within its Monte
Carlo error, and every deterministic step (the effective sample size, the
proposal covariance, the iteration and its restart, the error measure) is
tested against the R implementation.

<!-- python: run -->
```python
import math

import numpy as np
import pandas as pd

from quivers.formulas import PriorSpec, fit

rng = np.random.default_rng(1)
data = pd.DataFrame({"x": rng.normal(size=12)})
data["y"] = 0.5 + data["x"] + rng.normal(size=12)
priors = [
    PriorSpec(distribution="Normal(0.0, 1.0)", class_="Intercept"),
    PriorSpec(distribution="Normal(0.0, 1.0)", class_="b"),
]
full = fit(
    "y ~ x",
    data=data,
    fixed_parameters={"sigma": 1.0},
    priors=priors,
    method="nuts",
    num_warmup=100,
    num_samples=100,
    num_chains=2,
    seed=2,
)
estimate = full.bridge_sampler()
print(round(estimate.logml, 2), estimate.error_measures())

# With sigma fixed and Normal(0, 1) coefficients, y is Gaussian with
# covariance X X' + I, so the exact value is available.
X = np.column_stack([np.ones(12), data["x"]])
covariance = X @ X.T + np.eye(12)
y = data["y"].to_numpy()
exact = -0.5 * (
    12 * math.log(2 * math.pi)
    + np.linalg.slogdet(covariance)[1]
    + y @ np.linalg.solve(covariance, y)
)
assert abs(estimate.logml - exact) < 0.1
```

The SMC evidence ([`BayesianFit.evidence`][quivers.formulas._fit.BayesianFit.evidence])
estimates the same constant from prior to posterior by tempering. Both are
tested against closed-form and quadrature marginal likelihoods, and against
each other within their Monte Carlo errors.

### Bayes factors, posterior model probabilities, and model averaging

The [Bayes factor](https://en.wikipedia.org/wiki/Bayes_factor) of model 1
over model 2 is $p(y \mid M_1) / p(y \mid M_2)$ [@kass1995bayes], and the
posterior probability of model $k$ is proportional to
$p(y \mid M_k)\, p(M_k)$. [`bayes_factor`][quivers.formulas._evidence.bayes_factor]
and [`post_prob`][quivers.formulas._evidence.post_prob] accept fits, which
they bridge-sample, or stored [`BridgeResult`][quivers.diagnostics.bridge.BridgeResult]
estimates, and refuse fits of different response data, as brms does.
[`model_weights`](../api/formulas/fit.md#quivers.formulas.model_weights) gives
these probabilities with `weights="bma"`.

[`posterior_average`][quivers.formulas._model_averaging.posterior_average]
and [`pp_average`][quivers.formulas._model_averaging.pp_average] mix the
draws of several fits in proportion to model weights, which yields draws of
the mixture $\sum_k w_k\, p(\cdot \mid y, M_k)$ of
[Bayesian model averaging](https://en.wikipedia.org/wiki/Ensemble_learning#Bayesian_model_averaging)
[@hoeting1999bayesian] or of stacking [@yao2018stacking]. Model $k$
contributes the largest-remainder rounding of $w_k S$ of $S$ draws, chosen at
random without replacement. `posterior_average` mixes brms-named parameters
(`missing=` fills a parameter a model lacks), and `pp_average` mixes
predictions and summarizes them.

<!-- python: run -->
```python
from quivers.formulas import posterior_average, pp_average

reduced = fit("y ~ 1", data=data, method="svi", num_steps=50, seed=1)
wide = fit("y ~ x", data=data, method="svi", num_steps=50, seed=1)
models = {"wide": wide, "reduced": reduced}
mixed = posterior_average(models, weights=[0.3, 0.7], ndraws=100, seed=3)
print(mixed.ndraws, sorted(mixed.draws))
average = pp_average(models, weights="waic", method="posterior_epred", ndraws=50)
print(average.summary.values.shape)
```

### Automatic conditional effects

Called without a grid, [`BayesianFit.conditional_effects`][quivers.formulas._fit.BayesianFit.conditional_effects]
computes brms's automatic conditional effects. The default effects are every
population-level predictor variable and every pair of variables that appear
in one term, over all predicted parameters, with the variables of special
terms (`mo`, `me`, `mi`, `s`, `t2`, `gp`) crossed with each other. Each
effect is predicted on a grid: `resolution` equally spaced values over the
observed range of a numeric variable (unit steps for trial counts and
monotonic variables), the levels of a factor, and the mean and the mean plus
and minus one standard deviation of a numeric second variable (or its whole
range with `surface=True`, where `too_far` drops grid points far from the
data). Every other variable is held at a condition: the mean of a numeric
variable, the rounded median of an integer variable, the first level of a
factor, one for trial counts, and no group level (`re_formula="~0"`, brms's
`NA`). [`make_conditions`][quivers.formulas._brms_effects.make_conditions]
builds conditions over chosen variables, and `int_conditions` fixes the
values of effect variables. The result is a
[`BrmsConditionalEffects`][quivers.formulas._brms_effects.BrmsConditionalEffects]:
one labeled data frame per effect with brms's columns (`effect1__`,
`effect2__`, `cond__`, `estimate__`, `se__`, `lower__`, `upper__`), robust
summaries by default, the draws behind them, per-draw lines with
`spaghetti=True`, and the observed points assigned to conditions
(`select_points`). Columns of transformed terms that brms's frames carry,
such as `log(w)`, are not repeated, since the grid holds the variable `w`.

<!-- python: run -->
```python
from quivers.formulas import make_conditions

data["f"] = np.tile(["p", "q", "r"], 4)
grouped = fit("y ~ x * f", data=data, method="svi", num_steps=50, seed=1)
effects = grouped.conditional_effects(resolution=20)
print(effects.names())  # ('x', 'f', 'x:f')
frame = effects["x:f"].frame
print(frame[["x", "f", "estimate__", "lower__", "upper__"]].head(3))
by_level = grouped.conditional_effects("x", conditions=make_conditions(data, "f"))
print(by_level["x"].frame["cond__"].cat.categories.tolist())
```

`categorical=True` gives per-category probabilities of categorical and
ordinal models, with a `cats__` column (brms's `ordinal` is its deprecated
alias). `dpar` and `nlpar` predict a distributional or nonlinear parameter,
and `method` chooses `posterior_epred`, `posterior_linpred`, or
`posterior_predict`. A joint fit loops effects, then responses, and names
them `<resp>.<resp>_<effects>`. An explicit
[`prediction_grid`][quivers.diagnostics.conditional.prediction_grid] keeps
the bounded, empirical-reference effects described above.

`conditional_smooths()` evaluates the contribution of every `s()` and `t2()`
term on a grid of its covariates (a surface for two numeric covariates), with
`by` variables as conditions, as brms's `conditional_smooths` does. A joint
fit evaluates the smooths of every response, or of those named by `resp`,
and names each frame `<parameter>_<resp>: <term>`. The
function `quivers.formulas.conditional_smooths(fit, term, new_data)`
evaluates one smooth of a single-response fit at explicit rows. That call
takes the role of brms's `posterior_smooths(fit, smooth, newdata)`, a name
that is not provided.

### Posterior predictive checks

[`pp_check`][quivers.formulas._pp_check.pp_check] prepares the data behind
each bayesplot plot that brms forwards to [@gabry2019visualization]. It draws
replicated data (ten random draws, or every draw for the summary types, as
brms chooses), reads the observed responses and the `group` and `x`
variables, and removes censored responses (or keeps right-censored ones with
their event indicator for `km_overlay`). Leave-one-out types also smooth the
importance ratios by PSIS. The returned
[`PPCheck`][quivers.formulas._pp_check.PPCheck] holds bayesplot's table for
the type, computed by the `ppc_*` functions of
[`quivers.diagnostics`](../api/diagnostics/predictive_checks.md), which reproduce
bayesplot's tables: the long table of observed and replicated values, test
statistics, predictive intervals, errors and binned errors, frequency
tables, rootograms [@kleiber2016visualizing], LOO-PIT values, Kaplan-Meier
curves, and PIT ECDFs [@sailynoja2022graphical]. `prefix="ppd"` on a prior
view gives prior predictive checks.

Where ArviZ implements a counterpart, `PPCheck.plot(backend=...)` draws it
from `PPCheck.to_datatree()`: `dens_overlay`, `hist` and `ecdf_overlay` with
`plot_ppc_dist`, `stat` with `plot_ppc_tstat`, `intervals` and `ribbon` with
`plot_ppc_interval`, `bars` and `rootogram` with `plot_ppc_rootogram`,
`loo_pit*` with `plot_loo_pit`, `loo_intervals` and `loo_ribbon` with
`plot_loo_interval`, `pit_ecdf` with `plot_ppc_pit`, and `km_overlay` with
`plot_ppc_censored`. The other types (grouped variants, scatter, error,
`stat_2d`, violin, box, frequency polygon) have no ArviZ counterpart, so
their tables are returned for plotting with any library. bayesplot's
simultaneous bands of `pit_ecdf` are not computed; ArviZ's `plot_ppc_pit`
draws its own.

<!-- python: run -->
```python
check = grouped.pp_check("stat_grouped", group="f", stat="median")
print(check.data.head(4))
intervals = grouped.pp_check("intervals", x="x")
built = intervals.plot(backend="none")  # ArviZ builds the plot without drawing
```

### MCMC plots

[`mcmc_plot`][quivers.formulas._mcmc_plot.mcmc_plot] maps bayesplot's
`mcmc_*` types onto ArviZ plots of the brms-named draws, and `plot` and
`pairs` are its `combo` and `pairs` types, as in brms.

`variable` takes brms variable names or regular expressions. With
`regex=True` a pattern is searched for within each name, as brms's
`variable = "^b", regex = TRUE` is: `variable="^b_", regex=True` selects the
population coefficients and `variable="simo", regex=True` the simplexes of
monotonic terms. Without `regex` a pattern must match a whole name, as in
`variable="b_.*"`. A pattern matching no name raises an error.

| brms / bayesplot type | ArviZ counterpart |
| --- | --- |
| `intervals` | `plot_forest` |
| `areas`, `areas_ridges` | `plot_ridge` |
| `dens`, `dens_overlay`, `dens_chains`, `hist`, `hist_by_chain`, `violin` | `plot_dist` |
| `trace`, `trace_highlight` | `plot_trace` |
| `combo` (brms `plot`) | `plot_trace_dist` |
| `rank_hist`, `rank_overlay` | `plot_rank_dist` |
| `rank_ecdf` | `plot_rank` |
| `acf`, `acf_bar` | `plot_autocorr` |
| `neff`, `neff_hist` | `plot_ess` |
| `rhat`, `rhat_hist` | `plot_convergence_dist` |
| `pairs` (brms `pairs`), `scatter`, `hex` | `plot_pair` |
| `parcoord` | `plot_parallel` |
| `nuts_energy` | `plot_energy` |

The NUTS acceptance, divergence, step size, and tree depth plots have no
ArviZ counterpart, and their values are returned by `nuts_params()`. The
`recover_*` types need known true parameter values, which a fit does not
hold. brms's `stanplot` is a deprecated alias of `mcmc_plot` and is not
provided, and the ggplot themes (`theme_default`, `theme_black`) have no
meaning without ggplot.

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
morphism W : X -> Y [role=parameter, init=auto]
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
