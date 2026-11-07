# `quivers.diagnostics`

These adapters organize posterior draws and pointwise likelihoods for ArviZ. [Chain
diagnostics](../../guides/analysis-fitting-and-diagnostics.md#diagnostics-quiversdiagnostics) assess mixing and sampling precision. Predictive checks compare simulated
and observed data [@gelman2013bayesian], while [model comparison](../../tutorials/python/08-analysis-pipelines.md#model-comparison-with-psis-loo) estimates prediction on held-out observations.
Choose the analysis that addresses the discrepancy you are investigating.

Adapter to the ArviZ Bayesian-analysis ecosystem. Converts quivers'
inference records into `xarray.DataTree` objects consumable by
ArviZ functions that accept this container, including `plot_trace`,
`plot_forest`, `plot_ppc`, `loo`, `compare`, and `hdi`.

::: quivers.diagnostics
