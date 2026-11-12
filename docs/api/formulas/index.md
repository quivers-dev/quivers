# `quivers.formulas`

The formula front end compiles a
[regression formula](../../guides/analysis-data-and-formulas.md#formulas-quiversformulas) in the
style of brms [@burkner2017brms] to QVR source and
[fits](../../tutorials/python/08-analysis-pipelines.md#a-first-fit-gaussian-regression-in-one-line)
the resulting Bayesian model. In `y ~ x + (1 | group)`, `x` contributes a common slope and each
group receives its own intercept adjustment [@gelman2013bayesian]. The family and the priors
complete the statistical specification.

::: quivers.formulas
