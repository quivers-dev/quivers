# [Initialization](../../guides/analysis-fitting-and-diagnostics.md#recommend_init-and-apply_init_spec) recipes

Initialization chooses the starting values of trainable parameters. The reduction
algebra affects the scale of a composed result: summing many positive contributions
increases a sum-product weight, while noisy-OR approaches one. The recipes below account
for that difference when initializing parameters.

::: quivers.analysis.init_spec
