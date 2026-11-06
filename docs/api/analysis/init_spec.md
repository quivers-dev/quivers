# Initialization recipes

Initialization chooses the starting values of trainable parameters. The reduction
algebra affects the scale of a composed result: summing many positive contributions
increases a sum-product weight, while noisy-OR approaches one. The recipes below account
for that difference when initializing parameters.

See [applying an initialization recipe](../../guides/analysis-fitting-and-diagnostics.md#recommend_init-and-apply_init_spec) and
[inspecting the stages of a compiled program](../../tutorials/python/09-debugging.md#inspecting-a-compiled-program).

::: quivers.analysis.init_spec
