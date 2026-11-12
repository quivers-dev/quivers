# `quivers.formulas` fit entry point

`fit` ([one-line fit](../../guides/analysis-fitting-and-diagnostics.md#one-line-fit)) takes a formula, data, a family, and an inference configuration and returns a `BayesianFit`,
which gives [posterior summaries, predictive draws, and diagnostics](../../tutorials/python/08-analysis-pipelines.md#posterior-predictive-checks).
Posterior draws carry the uncertainty in the coefficients, and predictive draws add the variation
of the response given those coefficients. `formula_to_qvr` emits the QVR source of a formula.

::: quivers.formulas._fit
