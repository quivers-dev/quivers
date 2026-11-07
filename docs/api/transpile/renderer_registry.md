# Renderer registry

A [renderer registry](../../semantics/transpile-architecture.md#8-adding-a-new-backend) selects an emission function by backend and intermediate
representation node. Test each new emitter against the target’s syntax and the [semantic
checks](../../semantics/transpile-correctness/index.md#2-correctness-criterion) described in the linked guide.

The per-backend, per-node emit registry.

::: quivers.transpile.renderer_registry
