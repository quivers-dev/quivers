# `quivers.diagnostics.elpd`

An `ELPDEstimate` holds one estimate of the expected log pointwise predictive
density (ELPD) as the R `loo` package stores it [@vehtari2017loo]: the estimate,
the effective number of parameters, the deviance-scale criterion, their
standard errors, the pointwise terms, and Pareto diagnostics for PSIS-based
estimates [@vehtari2024psis]. ArviZ computes PSIS, comparisons, and stacking
weights [@yao2018stacking]. This module adds the difference estimator of
subsampled PSIS-LOO [@magnusson2020leave], PSIS-weighted expectations with
precomputed weights, and the leave-one-out R-squared of brms
[@gelman2019rsquared]. The [analysis guide](../../guides/analysis-fitting-and-diagnostics.md#leave-one-out-criteria)
shows the fit methods that produce these estimates.

::: quivers.diagnostics.elpd
