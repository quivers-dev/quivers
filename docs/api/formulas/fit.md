# `quivers.formulas` fit entry point

`fit` ([one-line fit](../../guides/analysis-fitting-and-diagnostics.md#one-line-fit)) takes a formula, data, a family, and an inference configuration and returns a `BayesianFit`,
which gives [posterior summaries, predictive draws, and diagnostics](../../tutorials/python/08-analysis-pipelines.md#posterior-predictive-checks).
Posterior draws carry the uncertainty in the coefficients, and predictive draws add the variation
of the response given those coefficients. `formula_to_qvr` emits the QVR source of a formula. `get_prior` returns the
effective named priors, including autoscaled defaults, without running inference.
Unknown or unused prior overrides raise `ValueError` before fitting.
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
conditional on each posterior draw. Its tensor has shape `(chain, draw, row)`;
categorical and ordinal responses use integer codes in `category_labels` order.
All three accept keyword-only `new_data` and `predictor_data`. New rows reuse
fitted transforms, contrasts, and known group levels; unseen categories and
groups are refused. Response and likelihood-weight columns are unnecessary,
but covariates, offsets, exposures, auxiliary predictors, and any per-row
trials must be supplied. Neural fits require explicit new `predictor_data`.
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

Trial-count families use `binomial_trials` for either one common count or a
column name. Trial counts satisfy `0 <= n <= 2**53`, including zero,
which permits only a zero response. This family-specific bound preserves
exact integer representation in native float64 arithmetic; `Int` itself
remains signed 64-bit. The same metadata and validation apply
to fitting and new-data prediction, including the zero-inflated binomial
and beta-binomial variants. See the
[count-family example](../../guides/analysis-data-and-formulas.md#zero-inflated-and-hurdle-counts).

::: quivers.formulas._fit
