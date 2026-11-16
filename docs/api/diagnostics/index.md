# `quivers.diagnostics`

The [convergence estimators](convergence.md) measure mixing and sampling precision for any tensor of
shape `(num_chains, num_draws, *event_shape)`, and
[simulation-based calibration](calibration.md) [@talts2018sbc] checks an inference procedure
against the program it targets. Both need only PyTorch. Predictive checks compare simulated and
observed data [@gelman2013bayesian], and
[model comparison](../../tutorials/python/08-analysis-pipelines.md#model-comparison-with-psis-loo)
estimates predictive accuracy on held-out observations. The ArviZ adapters convert inference
records of quivers into `xarray.DataTree` objects for the ArviZ functions that accept them,
including `plot_trace`, `plot_forest`, `plot_ppc`, `loo`, `compare`, and `hdi`, and need the
`diagnostics` extra.

[Shared posterior analysis](posterior.md) retains draws and raw response scores
from either source or formula fits. Its uncertainty summaries and empirical WAIC
need only PyTorch; MCMC convergence diagnostics require recorded chains.
DataTree conversion and PSIS-LOO use the optional ArviZ dependency.

::: quivers.diagnostics
