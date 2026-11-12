# Initialization recipes

These recipes choose the starting values of learnable parameters by `Algebra`
([`recommend_init` and `apply_init_spec`](../../guides/analysis-fitting-and-diagnostics.md#recommend_init-and-apply_init_spec)). The reduction
changes the scale of a composite: a sum over many positive terms grows, while noisy-OR approaches
one. Each recipe places the output of a composition of $k$ steps where gradients stay large. Under
noisy-OR, for instance, each entry starts near $\ln 2 / k$, so the composite starts near one half.

::: quivers.analysis.init_spec
