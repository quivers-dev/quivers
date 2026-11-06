# Renderer registry

A renderer registry selects an emission function by backend and intermediate
representation node. Test each new emitter against the target’s syntax and the semantic
checks described in the linked guide.

See [registering a backend](../../semantics/transpile-architecture.md#8-adding-a-new-backend) and
[the score-preservation criterion for a translation](../../semantics/transpile-correctness/index.md#2-correctness-criterion).

The per-backend, per-node emit registry.

::: quivers.transpile.renderer_registry
