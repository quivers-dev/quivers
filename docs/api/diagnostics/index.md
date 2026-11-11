# `quivers.diagnostics`

[Convergence estimators](convergence.md) assess mixing and sampling precision for any
`(num_chains, num_draws, *event_shape)` tensor and need only torch.
[Simulation-based calibration](calibration.md) checks an inference procedure against the program it
targets, and also needs only torch. The ArviZ adapters organize
posterior draws and pointwise likelihoods for ArviZ. Predictive checks compare simulated
and observed data [@gelman2013bayesian], while [model comparison](../../tutorials/python/08-analysis-pipelines.md#model-comparison-with-psis-loo) estimates prediction on held-out observations.
Choose the analysis that addresses the discrepancy you are investigating.

The adapters convert quivers' inference records into `xarray.DataTree`
objects consumable by ArviZ functions that accept this container,
including `plot_trace`, `plot_forest`, `plot_ppc`, `loo`, `compare`,
and `hdi`. They need the `diagnostics` extra.

::: quivers.diagnostics
