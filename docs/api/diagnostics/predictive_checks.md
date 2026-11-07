# `quivers.diagnostics.predictive_checks`

[Posterior predictive checks](../../tutorials/python/08-analysis-pipelines.md#posterior-predictive-checks) compare a feature of the observations with the same feature
in replicated data [@gelman2013bayesian]. For instance, an observed upper quantile outside the range of
simulated upper quantiles can expose a poorly modeled tail. Leave-one-out probability
integral transform (LOO-PIT) checks assess calibration of predictions for held-out
observations.

Posterior-predictive p-values and LOO-PIT calibration.

::: quivers.diagnostics.predictive_checks
