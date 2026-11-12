# Kernel IR

`IRQiecModule` is the typed [IR](../../semantics/transpile-architecture.md#2-the-ir) of a checked QIEC
module, independent of any target. It keeps the computations and their dependencies, so a renderer
can inspect the effects and indexed operations of the reachable program.

::: quivers.transpile.qiec_ir
