# `quivers.diagnostics.predictive_checks`

Posterior predictive checks compare a feature of the observations with the same feature
in replicated data. For instance, an observed upper quantile outside the range of
simulated upper quantiles can expose a poorly modeled tail. Leave-one-out probability
integral transform (LOO-PIT) checks assess calibration of predictions for held-out
observations.

See [a posterior predictive check](../../tutorials/python/08-analysis-pipelines.md#posterior-predictive-checks) and
[checking the posterior draws used by these analyses](../../tutorials/qvr/08-diagnostics-and-comparison.md#convergence-diagnostics).

For Bayesian model checking and predictive interpretation, see [@gelman2013bayesian].

Posterior-predictive p-values and LOO-PIT calibration.

::: quivers.diagnostics.predictive_checks
