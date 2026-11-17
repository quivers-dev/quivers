# `quivers.diagnostics.comparison`

[Leave-one-out (LOO) comparison](../../tutorials/python/08-analysis-pipelines.md#model-comparison-with-psis-loo)
estimates how well each observation is predicted by a fit to the others [@vehtari2017loo].
Pareto-smoothed importance sampling (PSIS) approximates it from
[posterior draws](../../tutorials/qvr/08-diagnostics-and-comparison.md#convergence-diagnostics) and
flags unstable importance weights [@vehtari2024psis], and stacking chooses weights for combining the
predictions of several models [@yao2018stacking]. The module computes pointwise PSIS-LOO and model
comparisons through ArviZ. `compare` accepts a mapping of model names to
`PosteriorAnalysis` objects or DataTrees. Retained analyses reuse their raw
likelihoods; compared analyses must agree on score units and observed response
values. Variational analyses retain their provenance and use relative efficiency
one, without correcting variational approximation error. The
[shared analysis API](posterior.md) describes the retained tensors.

::: quivers.diagnostics.comparison
