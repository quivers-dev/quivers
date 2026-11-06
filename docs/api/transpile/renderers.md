# Renderers

Renderers emit source for a specific backend from the common representation. They
translate sampling operations, observations, and density contributions into the target’s
conventions. To check a translation, fix the data and latent values and compare the
resulting model scores.

See [rendering target source](../../semantics/transpile-architecture.md#5-renderert-ir-panprotoschema) and
[the score-preservation criterion for a translation](../../semantics/transpile-correctness/index.md#2-correctness-criterion).

The per-backend renderers consuming the IR.

::: quivers.transpile.renderers
