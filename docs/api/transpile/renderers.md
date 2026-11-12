# Renderers

Each [renderer](../../semantics/transpile-architecture.md#5-renderert-ir-panprotoschema) emits source
for one backend from the IR, translating draws, observations, and density terms into the target's
conventions. A translation is [checked](../../semantics/transpile-correctness/index.md#2-correctness-criterion)
by fixing the data and latent values and comparing the log densities of the source and target
models.

::: quivers.transpile.renderers
