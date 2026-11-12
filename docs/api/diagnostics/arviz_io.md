# `quivers.diagnostics.arviz_io`

These functions convert an `MCMCResult`, stochastic variational inference (SVI) draws, or predictive draws into an `xarray.DataTree`
for ArviZ ([diagnostics](../../guides/analysis-fitting-and-diagnostics.md#diagnostics-quiversdiagnostics)).
Named dimensions separate chains, draws, observations, and the components of one value, and latent
coefficients and predicted responses go to [their own groups](../../tutorials/qvr/08-diagnostics-and-comparison.md#the-datatree-bridge).

::: quivers.diagnostics.arviz_io
