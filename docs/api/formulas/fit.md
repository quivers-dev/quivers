# `quivers.formulas` fit entry point

`fit` ([one-line fit](../../guides/analysis-fitting-and-diagnostics.md#one-line-fit)) takes a formula, data, a family, and an inference configuration and returns a `BayesianFit`,
which gives [posterior summaries, predictive draws, and diagnostics](../../tutorials/python/08-analysis-pipelines.md#posterior-predictive-checks).
Posterior draws carry the uncertainty in the coefficients, and predictive draws add the variation
of the response given those coefficients. `formula_to_qvr` emits the QVR source of a formula. `get_prior` returns the
effective named priors, including autoscaled defaults, without running inference.
Unknown or unused prior overrides raise `ValueError` before fitting.
`sample_prior="yes"` retains an independent prior stream without changing the
fitted posterior; `sample_prior="only"` disables observation likelihoods during
the selected inference method and retains priors and explicit score potentials.
`prior_draws` and `prior_view` reuse the retained conditional laws, data-dependent
priors, encodings, and existing prediction/likelihood methods. Independent
samples have no MCMC convergence statistics. See
[prior-only fitting](../../guides/analysis-fitting-and-diagnostics.md#prior-only-fitting-and-independent-prior-draws)
for seed and factor-conditioned sampling semantics.
`fixed_parameters={"quantile": 0.9}` performs asymmetric Laplace quantile regression
at a specified quantile, and other auxiliary parameters can likewise be held fixed.
All three entry points accept
`distributional={"sigma": "~ z"}` to give an auxiliary parameter its own predictor;
[distributional regression](../../guides/analysis-data-and-formulas.md#distributional-regression-and-link-selection)
describes coefficient names, priors, and group coding.

`BayesianFit.draws` preserves chain and draw axes. `fixed_effects`, `random_effects`,
and `coefficients` expose population and group coefficients with retained labels.
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

::: quivers.formulas._fit
