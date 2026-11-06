# `quivers.diagnostics.arviz_io`

This adapter organizes inference and predictive results into ArviZ `DataTree` groups.
Named dimensions distinguish chains, draws, observations, and the components of one
value. Assign latent coefficients and predicted responses to their corresponding groups
so diagnostics operate on the intended quantities.

See [exporting results to ArviZ](../../guides/analysis-fitting-and-diagnostics.md#diagnostics-quiversdiagnostics) and
[assigning posterior and predictive groups](../../tutorials/qvr/08-diagnostics-and-comparison.md#the-datatree-bridge).

For Bayesian model checking and predictive interpretation, see [@gelman2013bayesian].

Conversion from `MCMCResult`, SVI draws, or predictive draws to
`xarray.DataTree`.

::: quivers.diagnostics.arviz_io
