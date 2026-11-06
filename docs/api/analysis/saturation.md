# Saturation diagnostics

Saturation occurs when bounded values approach an endpoint and their gradients become
small. These diagnostics report where it occurs in a composition. Inspect the parameter
initialization and the number of contributions to a reduction when many values approach
zero or one.

See [locating saturated values](../../guides/analysis-fitting-and-diagnostics.md#saturation_warnings) and
[inspecting the stages of a compiled program](../../tutorials/python/09-debugging.md#inspecting-a-compiled-program).

::: quivers.analysis.saturation
