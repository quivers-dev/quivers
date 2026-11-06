# `quivers.formulas` fit entry point

The fit entry point combines a formula, data, family, and inference configuration. Its
`BayesianFit` result supports posterior summaries, predictive simulations, and
diagnostics. Posterior draws summarize uncertainty in coefficients; predictive
simulations also include response variation conditional on those coefficients.

See [fitting a formula](../../guides/analysis-fitting-and-diagnostics.md#one-line-fit) and
[checking its predictions](../../tutorials/python/08-analysis-pipelines.md#posterior-predictive-checks).

For regression and partial pooling, see [@gelman2013bayesian].

The user-facing `fit` entry point, the `formula_to_qvr` source emitter,
and the `BayesianFit` result wrapper.

::: quivers.formulas._fit
