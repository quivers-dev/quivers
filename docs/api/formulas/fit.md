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
`posterior_linpred` and `posterior_epred` replay native fitted bindings on the
training rows, including native class probabilities and expected counts. Their
[posterior workflow](../../guides/analysis-data-and-formulas.md#posterior-coefficients-and-predictions)
describes shapes, labels, link scales, and expectations.

::: quivers.formulas._fit
