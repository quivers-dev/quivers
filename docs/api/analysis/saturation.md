# Saturation diagnostics

A bounded value *saturates* when it approaches an endpoint of its range and its gradient
vanishes. `SaturationWarning`s
([`saturation_warnings`](../../guides/analysis-fitting-and-diagnostics.md#saturation_warnings)) report, at their source locations, the steps where an `Algebra`'s
initialization recipe differs from the default `Normal(0, 1)` enough to predict saturation, such as
a deep chain under noisy-OR, which approaches one in every entry
([initialization](../../guides/analysis-fitting-and-diagnostics.md#recommend_init-and-apply_init_spec)).

::: quivers.analysis.saturation
