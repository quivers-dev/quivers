# `quivers.diagnostics`

These adapters organize posterior draws and pointwise likelihoods for ArviZ. Chain
diagnostics assess mixing and sampling precision. Predictive checks compare simulated
and observed data, while model comparison estimates prediction on held-out observations.
Choose the analysis that addresses the discrepancy you are investigating.

See [the diagnostics interface](../../guides/analysis-fitting-and-diagnostics.md#diagnostics-quiversdiagnostics) and
[assigning posterior and predictive groups](../../tutorials/qvr/08-diagnostics-and-comparison.md#the-datatree-bridge).

For Bayesian model checking and predictive interpretation, see [@gelman2013bayesian].

Adapter to the ArviZ Bayesian-analysis ecosystem. Converts quivers'
inference records into `xarray.DataTree` objects consumable by
ArviZ functions that accept this container, including `plot_trace`,
`plot_forest`, `plot_ppc`, `loo`, `compare`, and `hdi`.

::: quivers.diagnostics
