# `quivers.diagnostics.comparison`

[Leave-one-out (LOO) comparison](../../tutorials/python/08-analysis-pipelines.md#model-comparison-with-psis-loo) estimates how well each observation would be predicted
after fitting to the other observations [@vehtari2017loo]. Pareto-smoothed importance sampling (PSIS)
approximates this calculation from [posterior draws](../../tutorials/qvr/08-diagnostics-and-comparison.md#convergence-diagnostics) and reports unstable importance
weights [@vehtari2024psis]. Stacking uses predictive performance to choose weights for combining model
predictions [@yao2018stacking].

Pointwise PSIS-LOO and model comparison via ArviZ.

::: quivers.diagnostics.comparison
