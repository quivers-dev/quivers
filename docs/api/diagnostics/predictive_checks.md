# `quivers.diagnostics.predictive_checks`

A [posterior predictive check](../../tutorials/python/08-analysis-pipelines.md#posterior-predictive-checks)
compares a statistic of the observations with its distribution over replicated data
[@gelman1996predictive; @gelman2013bayesian]. An observed upper quantile beyond the range of the
replicated ones, for instance, exposes a poorly modeled tail. The module gives posterior predictive
p-values and leave-one-out probability integral transform (LOO-PIT) checks of calibration for
held-out observations.

::: quivers.diagnostics.predictive_checks

## The data behind bayesplot's checks

The `ppc_*` functions of [`quivers.diagnostics`](index.md) compute the tables bayesplot draws its
`ppc_*` plots from [@gabry2019visualization], with bayesplot's column names
and row order. `pp_check` on a fit selects the draws and data as brms does
(see the [analysis guide](../../guides/analysis-fitting-and-diagnostics.md#posterior-predictive-checks)).

::: quivers.diagnostics._ppc_data
