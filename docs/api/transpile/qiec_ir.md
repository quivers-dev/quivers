# Kernel IR

This [typed intermediate representation](../../semantics/transpile-architecture.md#2-the-ir) carries the checked core into transpilation. It
retains computations and their dependencies so renderers can inspect the effects and
indexed operations of the reachable program.

The typed, target-independent mirror of a checked kernel module.

::: quivers.transpile.qiec_ir
