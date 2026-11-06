# `quivers.diagnostics.comparison`

Leave-one-out (LOO) comparison estimates how well each observation would be predicted
after fitting to the other observations. Pareto-smoothed importance sampling (PSIS)
approximates this calculation from posterior draws and reports unstable importance
weights. Stacking uses predictive performance to choose weights for combining model
predictions.

See [comparing models with PSIS-LOO](../../tutorials/python/08-analysis-pipelines.md#model-comparison-with-psis-loo) and
[checking the posterior draws used by these analyses](../../tutorials/qvr/08-diagnostics-and-comparison.md#convergence-diagnostics).

For practical leave-one-out comparison and its diagnostics, see [@vehtari2017loo].

Pointwise PSIS-LOO and model comparison via ArviZ.

Sources for the constructions below: [@yao2018stacking; @vehtari2024psis].

::: quivers.diagnostics.comparison
