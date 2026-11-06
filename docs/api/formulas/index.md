# `quivers.formulas`

A regression formula describes how a response depends on predictors. In `y ~ x + (1 |
group)`, `x` contributes a common slope and each group receives an intercept adjustment.
This frontend turns that description into a QVR model and runs inference; the family and
priors complete the statistical specification.

See [formula syntax](../../guides/analysis-data-and-formulas.md#formulas-quiversformulas) and
[a Gaussian regression fit](../../tutorials/python/08-analysis-pipelines.md#a-first-fit-gaussian-regression-in-one-line).

For regression and partial pooling, see [@gelman2013bayesian].

A brms-style formula frontend over the QVR DSL. It compiles a
regression formula to `.qvr` source and fits the resulting Bayesian
model.

::: quivers.formulas
