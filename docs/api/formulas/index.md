# `quivers.formulas`

A [regression formula](../../guides/analysis-data-and-formulas.md#formulas-quiversformulas) describes how a response depends on predictors. In `y ~ x + (1 |
group)`, `x` contributes a common slope and each group receives an intercept adjustment [@gelman2013bayesian].
This frontend turns that description into a QVR model and [runs inference](../../tutorials/python/08-analysis-pipelines.md#a-first-fit-gaussian-regression-in-one-line); the family and
priors complete the statistical specification.

A brms-style formula frontend over the QVR DSL. It compiles a
regression formula to `.qvr` source and fits the resulting Bayesian
model.

::: quivers.formulas
