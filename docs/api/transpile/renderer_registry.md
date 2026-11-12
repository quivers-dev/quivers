# Renderer registry

The [renderer registry](../../semantics/transpile-architecture.md#8-adding-a-new-backend) selects an
emit function by backend and IR node. A new emitter is tested against the target's syntax and
against the [correctness criterion](../../semantics/transpile-correctness/index.md#2-correctness-criterion),
equal log densities at fixed values.

::: quivers.transpile.renderer_registry
