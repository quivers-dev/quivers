# `quivers.diagnostics.arviz_io`

This adapter [organizes inference and predictive results](../../guides/analysis-fitting-and-diagnostics.md#diagnostics-quiversdiagnostics) into ArviZ `DataTree` groups.
Named dimensions distinguish chains, draws, observations, and the components of one
value. Assign latent coefficients and predicted responses to their [corresponding groups](../../tutorials/qvr/08-diagnostics-and-comparison.md#the-datatree-bridge)
so diagnostics operate on the intended quantities.

Conversion from `MCMCResult`, SVI draws, or predictive draws to
`xarray.DataTree`.

::: quivers.diagnostics.arviz_io
