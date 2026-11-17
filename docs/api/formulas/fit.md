# `quivers.formulas` fit entry point

`fit` ([one-line fit](../../guides/analysis-fitting-and-diagnostics.md#one-line-fit)) takes a formula, data, a family, and an inference configuration and returns a `BayesianFit`,
which gives [posterior summaries, predictive draws, and diagnostics](../../tutorials/python/08-analysis-pipelines.md#posterior-predictive-checks).
Posterior draws carry the uncertainty in the coefficients, and predictive draws add the variation
of the response given those coefficients. `formula_to_qvr` emits the QVR source of a formula. `get_prior` returns the
effective named priors, including autoscaled defaults, without running inference.
Unknown or unused prior overrides raise `ValueError` before fitting.
`BayesianFit.update(data=None, **overrides)` reuses a retained fitting recipe
with the original raw dataframe and all canonical fit settings. It fits a fresh
posterior, with isolated initial neural state and explicit clone/reference data
ownership. Method changes clear inherited inactive budgets; explicit invalid
overrides still fail. See [updating and refitting](../../guides/model-refits.md)
for data ownership and adapters to the shared cross-validation runner.
`BayesianFit.refit_callback(method_factory=..., num_samples=...)` supplies that
runner with fresh formula fits and raw held-out likelihoods. Fold coordinates
address retained included rows; the adapter maps them to original rows and
partitions retained neural inputs by the same positions. Its method factory
owns fitting seeds and budgets, while prediction keywords retain the existing
group-selection and new-level policies. See the
[formula partition example](../../guides/model-refits.md#formula-models).
`sample_prior="yes"` retains an independent prior stream without changing the
fitted posterior; `sample_prior="only"` disables observation likelihoods during
the selected inference method and retains priors and explicit score potentials.
`prior_draws` and `prior_view` reuse the retained conditional laws, data-dependent
priors, encodings, and existing prediction/likelihood methods. Independent
samples have no MCMC convergence statistics. See
[prior-only fitting](../../guides/analysis-fitting-and-diagnostics.md#prior-only-fitting-and-independent-prior-draws)
for seed and factor-conditioned sampling semantics.
`BayesianFit.evidence(sampler)` applies the existing
[`SMC` driver](../inference/mcmc.md#quivers.inference.smc.SMC) to the retained program and training data,
without recompilation or conversion of the fitted posterior. It returns
`SMCResult`, including particles, the temperature ladder, and `log_evidence`.
For instance:

```python
import pandas as pd
from quivers.formulas import fit
from quivers.inference import SMC
from quivers.inference.mcmc import MetropolisKernel, RandomWalkProposal

fitted = fit(
    "y ~ 1", data=pd.DataFrame({"y": [0.2, 0.7, 1.1]}),
    fixed_parameters={"sigma": 1.0}, method="svi", num_steps=1, seed=7,
)
evidence = fitted.evidence(SMC(
    MetropolisKernel(RandomWalkProposal(0.5)), num_particles=128,
    num_mcmc_steps=2, num_pilot_steps=0, seed=11,
))
assert evidence.num_particles == 128
```

This is a separate inference run whose budget and seed belong to `sampler`.
Its log normalizer is ordinary marginal data evidence when the retained prior
and likelihood form a normalized sampling model. Non-unit response weights
instead yield the powered target's log normalizer. SMC retains its existing
refusals for unknown prior mass, opaque joint overrides, and subsampled
observations; the adapter does not bypass them.
`fixed_parameters={"quantile": 0.9}` performs asymmetric Laplace quantile regression
at a specified quantile, and other auxiliary parameters can likewise be held fixed.
All three entry points accept
`distributional={"sigma": "~ z"}` to give an auxiliary parameter its own predictor;
[distributional regression](../../guides/analysis-data-and-formulas.md#distributional-regression-and-link-selection)
describes coefficient names, priors, and group coding.

`BayesianFit.draws` preserves chain and draw axes. `fixed_effects`, `random_effects`,
and `coefficients` expose population and group coefficients with retained labels.
`hypothesis("x - Intercept")` evaluates a checked QVR contrast over the same
population coefficients. `parameter` selects a distributional or nonlinear
predictor, `class_` selects an emitted parameter role, and `scope="ranef"`
or `"coef"` with `group` retains the level axis of adjustments or combined
coefficients. `class_=None` uses exact latent site names. `prior=True` evaluates
the contrast on the corresponding joint prior stream, yielding region evidence
in addition to posterior odds. It does not estimate point-null density evidence.
See [formula hypotheses](../../guides/analysis-data-and-formulas.md#checked-hypotheses-over-fitted-coefficients)
and [contrast summaries](../diagnostics/hypothesis.md).
An explicit `point=PointNullSpec(...)` with `prior=True` also returns
`point_evidence` from those same joint streams. Its marginal density ratio,
optional declared-assumption point Bayes factor, bandwidths, and numerical
limits follow the [shared density interface](../diagnostics/density.md).

`posterior_linpred`, `posterior_epred`, and `posterior_predict` replay native
fitted bindings on the training rows or on a supplied `new_data` frame. The first two expose predictors and
response expectations; `posterior_predict` independently samples responses
conditional on each posterior draw. Its tensor has shape `(chain, draw, row)`
followed by any response event dimensions; categorical and ordinal responses
use integer codes in `category_labels` order.
`posterior_thresholds` additionally exposes effective ordinal cutpoints as
`(chain, draw, row, threshold)`, labeled by adjacent category pairs in
`threshold_labels`. It excludes the proportional population predictor
and discrimination. All four accept keyword-only `new_data` and `predictor_data`. New rows reuse
fitted transforms, contrasts, and known group levels. Ordinary categorical
predictor levels remain strict; `allow_new_levels=True` admits new or
missing grouping identities. `sample_new_levels` selects per-draw donors
(`"uncertainty"`), one donor index across draws (`"old_levels"`), or
the fitted Normal/LKJ group law (`"gaussian"`). A grouped Cox baseline
replays its Dirichlet profile law under this policy. Donors are joint across
slopes, auxiliary predictors, and baseline profiles. For constrained cumulative category-specific
new-group profiles or effects, explicit `"feasible_uncertainty"` and
`"feasible_gaussian"` condition the joint proposal on requested-row
ordering. They retain all posterior draws and refuse after 1,024 failed
proposals. Fitted cutpoints are replayed without reconstruction from fresh
rows or selected effects; crossing profiles fail with located diagnostics.
`re_formula=None` retains fitted varying
effects, `"~0"` sets them to zero, and a one-sided group formula selects
fitted terms. Unsupported selectors fail explicitly. Response and
likelihood-weight columns are unnecessary, but retained covariates,
offsets, exposures, auxiliary predictors, and per-row trials must be
supplied, together with retained measurement-error and truncation columns
when those sampling-law roles use data columns. Censoring and training
subset columns are unnecessary. Neural fits require explicit new `predictor_data`.
Training predictions reuse the compiled program; new rows specialize one
program per request. Both fill one preallocated output tensor across draws.
The [posterior workflow](../../guides/analysis-data-and-formulas.md#posterior-coefficients-and-predictions)
describes shapes, labels, link scales, and expectations.
The [new-row workflow](../../guides/analysis-data-and-formulas.md#predictions-on-new-rows)
gives an executable example and the retained evaluator's serialization boundary.

`BayesianFit.bayes_r2` reuses these prediction bindings to report per-draw
regression variance ratios on the response scale. `method="model"` uses average
native conditional variance; `method="residual"` uses actual-minus-expected
response variance. Auto uses model variances for ordinary Gaussian/Bernoulli
responses and residual variances otherwise; explicit observations select the
residual definition. New-data residual calculations evaluate the fitted response
transform or accept `observed` directly. Censoring bounds cannot substitute for
actual responses. Ordered categories use their retained integer scores, while
unordered categories require a chosen metric. See
[regression variance ratios](../../guides/analysis-data-and-formulas.md#regression-variance-ratios)
for the executable workflow and [the pure diagnostic](../diagnostics/regression.md)
for scaling, event axes, and degenerate cases.

`fit`, `formula_to_qvr`, and `get_prior` accept `weights="column"` for
observation-only likelihood weights. The column is retained on the parsed
formula; posterior predictors use the unweighted response law at each draw.
See [likelihood weights](../../guides/analysis-data-and-formulas.md#likelihood-weights).

Both `reparameterize="centered"` and `"noncentered"` support correlated
blocks with mixed scalar, transition, and response-component widths.
Coefficient labels and inspection methods retain their meanings across
coordinates. Correlated centered blocks have a joint
`group_coefficients_<block>` site; centered grouped threshold log slacks
use `cutpoint_log_spacing_value`. These are natural-scale site names for
`initial_values`; the corresponding noncentered sites contain innovations.
Gaussian prediction prepares the requested group sites' upstream prior
program once per request and replays their conditional law at each draw,
without executing unrelated downstream response computations.

## Nonlinear predictors

`nonlinear` maps parameter names to one-sided additive formulas or
`NonlinearExpression` values. With the argument present, the main right
side is a deterministic QVR expression; `distributional` expressions supply
other family outputs on their link scales. The graph is dependency-checked,
retains each additive leaf's fitted design, and applies each output's
inverse link once. Component mappings use explicit free-coordinate labels.
The [nonlinear workflow](../../guides/analysis-data-and-formulas.md#nonlinear-parameter-predictors)
gives an executable fit and explains coefficient names and default priors.

::: quivers.formulas.NonlinearExpression

::: quivers.formulas.ComponentExpression

::: quivers.formulas.PredictorGraph

## Decoded response draws

`posterior_response` returns `ResponseDraws` after one posterior predictive
replay. Its `value` tensor preserves the response's measurement coordinates;
for a Wiener response, this is positive reaction time including nondecision
delay, and `decision` records zero for the lower boundary and one for the
upper boundary. `decision_labels` names those codes. `posterior_predict`
retains the corresponding lossless signed response tensor.

`PredictionOptions` defines the shared keyword contract for these two
prediction methods. `ExpectationOptions` additionally includes `mc_samples`
for explicitly requested Monte Carlo expectations. `posterior_choice_probs`
uses the same expectation keywords and returns the joint model's lower/upper
choice probabilities; it does not estimate them by repeated response sampling.

::: quivers.formulas.ResponseDraws

::: quivers.formulas.PredictionOptions

::: quivers.formulas.ExpectationOptions

## Likelihoods and analysis

`BayesianFit.log_likelihood(num_samples=100, seed=7)` returns raw response log
densities or recorded-event log probabilities on training rows, with chain
and draw axes followed by the observation axis. Truncation normalizes the
response law before the event is scored. These scores exclude fitting
weights and priors. `analysis` retains the
same posterior draws and scores for summaries, empirical WAIC, DataTree
conversion, and PSIS-LOO. It samples a variational guide once per request and
never treats its pseudo-chain as recorded MCMC output.
The [executable workflow](../../guides/analysis-data-and-formulas.md#raw-likelihoods-and-retained-analysis)
and [shared API](../diagnostics/posterior.md) describe score units, optional
ArviZ dependencies, and the distinction between uncertainty and convergence.

## Smooth contributions

`conditional_smooths(fitted, term, new_data)` returns one fitted `s()`,
`t2()`, or approximate `gp()` term's posterior contribution to its linear predictor at explicit
rows, as brms's `conditional_smooths` reports it. The
[structured predictor guide](../../guides/formula-special-predictors.md)
states the bases and their retained prediction state.

::: quivers.formulas.conditional_smooths

## Named initial values

`fit(..., initial_values={...})` initializes latent sites by their generated
QVR names, which are visible in `formula_to_qvr` output and `get_prior` results.
Values use each site's natural scale; scalars broadcast and tensors must match
the site's full declared shape. Unknown names, observed sites, Boolean or
nonfinite values, incompatible shapes, and support boundaries are refused.
The inference object detaches and copies the supplied values.

```python
import pandas as pd

from quivers.formulas import fit

result = fit(
    "y ~ x",
    data=pd.DataFrame({"y": [0.1, 0.4, 0.8], "x": [-1.0, 0.0, 1.0]}),
    method="svi",
    num_steps=10,
    seed=2026,
    initial_values={"intercept": 0.4, "beta_x": 0.0, "sigma": 0.5},
)
assert set(result.draws(num_samples=2)) == {"intercept", "beta_x", "sigma"}
```

For `method="svi"`, the default `AutoNormalGuide` maps these values into its
[initial unconstrained centers](../../guides/inference-svi.md#named-initial-centers).
A custom guide must accept `initial_values` when a nonempty mapping is supplied.
For `"hmc"` and `"nuts"`, named values replace the corresponding coordinates
after seeded initialization, as in the
[MCMC interface](../../guides/inference-mcmc.md#named-initial-values).

A `shifted_lognormal` fit with a regressed `ndt`, a log link, and an intercept
initializes the population intercept to `log(min(y) / 4)` and the other
population coefficients to zero. The implied population shift thus starts
below every training response. Group effects retain the inference method's
starting values. These are starting coefficients; the model's predictor and
priors remain unchanged. Explicit `initial_values` override the automatic
entries and can also set group sites. A custom SVI guide used with this
automatic initialization must also accept the mapping.

## Hazard rate and expected event time

For a `cox` fit, `posterior_linpred(transform=True)` returns the positive
hazard multiplier. `posterior_predict` draws positive event times from
the retained spline baseline and its explicit constant-hazard tail.
`posterior_epred(mc_samples=...)` approximates expected event time using
the shared finite-mean sampler; omitting the budget fails explicitly.
The fitted knots and coefficient table are retained on new predictor rows.
See the [Cox example](../../guides/analysis-data-and-formulas.md#spline-proportional-hazards).

## Trial counts

Trial-count families use `binomial_trials` for either one common count or a
column name. Trial counts satisfy `0 <= n <= 2**53`, including zero,
which permits only a zero response. This family-specific bound preserves
exact integer representation in native float64 arithmetic; `Int` itself
remains signed 64-bit. The same metadata and validation apply
to fitting and new-data prediction, including the zero-inflated binomial
and beta-binomial variants. See the
[count-family example](../../guides/analysis-data-and-formulas.md#zero-inflated-and-hurdle-counts).

## brms post-processing

`BayesianFit` and `MultivariateFit` carry brms's summary, residual, accessor,
and leave-one-out methods: `summary`, `fixef`, `ranef`, `coef`, `varcorr`,
`vcov`, `variables`, `posterior_summary`, `posterior_interval`, `predict`,
`fitted`, `residuals`, `predictive_error`, `predictive_interval`, `ndraws`,
`nchains`, `niterations`, `nobs`, `ngrps`, `rhat`, `neff_ratio`,
`nuts_params`, `log_posterior`, `loo`, `loo_moment_match`, `loo_subsample`,
`waic`, `psis`, `loo_predict`, `loo_epred`, `loo_linpred`,
`loo_predictive_interval`, `loo_r2`, `kfold`, and `add_criterion`. The
[analysis guide](../../guides/analysis-fitting-and-diagnostics.md#brms-summaries-residuals-and-predictive-criteria)
states their brms counterparts and the deliberate differences.

::: quivers.formulas.FitSummary

::: quivers.formulas.VarCorrBlock

::: quivers.formulas.KFoldResult

::: quivers.formulas.KFoldPrediction

::: quivers.formulas.SinglePredictionOptions

::: quivers.formulas.JointPredictionOptions

::: quivers.formulas.kfold_predict

::: quivers.formulas.loo_compare

::: quivers.formulas.loo_model_weights

::: quivers.formulas.model_weights

::: quivers.formulas._postprocess_methods.SinglePostprocess

::: quivers.formulas._postprocess_methods.JointPostprocess

## brms evidence, effects, and predictive checks

`BayesianFit` and `MultivariateFit` carry brms's marginal-likelihood,
conditional-effects, predictive-check and MCMC-plot methods:
`bridge_sampler`, `bayes_factor`, automatic `conditional_effects`,
`conditional_smooths`, `pp_check`, `mcmc_plot`, `plot`, and `pairs`. The
functions `post_prob`, `posterior_average`, `pp_average`, and
`make_conditions` take several fits or a data frame. The
[analysis guide](../../guides/analysis-fitting-and-diagnostics.md#evidence-effects-and-predictive-checks)
states their brms counterparts and the deliberate differences.

::: quivers.formulas.bridge_sampler

::: quivers.formulas.bayes_factor

::: quivers.formulas.post_prob

::: quivers.formulas.BayesFactor

::: quivers.formulas.posterior_average

::: quivers.formulas.pp_average

::: quivers.formulas.PosteriorAverage

::: quivers.formulas.PredictionAverage

::: quivers.formulas.make_conditions

::: quivers.formulas.BrmsConditionalEffects

::: quivers.formulas.EffectFrame

::: quivers.formulas.pp_check

::: quivers.formulas.PPCheck

::: quivers.formulas.mcmc_plot

::: quivers.formulas._evidence

::: quivers.formulas._model_averaging

::: quivers.formulas._brms_effects

::: quivers.formulas._pp_check

::: quivers.formulas._mcmc_plot

::: quivers.formulas._evidence_effects.EvidenceEffectsMethods

::: quivers.formulas._fit
