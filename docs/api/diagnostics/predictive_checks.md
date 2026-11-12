# `quivers.diagnostics.predictive_checks`

A [posterior predictive check](../../tutorials/python/08-analysis-pipelines.md#posterior-predictive-checks)
compares a statistic of the observations with its distribution over replicated data
[@gelman1996predictive; @gelman2013bayesian]. An observed upper quantile beyond the range of the
replicated ones, for instance, exposes a poorly modeled tail. The module gives posterior predictive
p-values and leave-one-out probability integral transform (LOO-PIT) checks of calibration for
held-out observations.

::: quivers.diagnostics.predictive_checks
