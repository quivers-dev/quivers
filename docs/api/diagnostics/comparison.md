# `quivers.diagnostics.comparison`

[Leave-one-out (LOO) comparison](../../tutorials/python/08-analysis-pipelines.md#model-comparison-with-psis-loo)
estimates how well each observation is predicted by a fit to the others [@vehtari2017loo].
Pareto-smoothed importance sampling (PSIS) approximates it from
[posterior draws](../../tutorials/qvr/08-diagnostics-and-comparison.md#convergence-diagnostics) and
flags unstable importance weights [@vehtari2024psis], and stacking chooses weights for combining the
predictions of several models [@yao2018stacking]. The module computes pointwise PSIS-LOO and model
comparisons through ArviZ.

::: quivers.diagnostics.comparison
