# [Saturation](../../guides/analysis-fitting-and-diagnostics.md#saturation_warnings) diagnostics

Saturation occurs when bounded values approach an endpoint and their gradients become
small. These diagnostics report where it occurs in a composition. Inspect the [parameter
initialization](../../guides/analysis-fitting-and-diagnostics.md#recommend_init-and-apply_init_spec) and the number of contributions to a reduction when many values approach
zero or one.

::: quivers.analysis.saturation
