# Lowering support

Lowering uses shape, support, and family information to express a model in the
representation consumed by target renderers. These support tables guide that process.
They are implementation details; use the public transpilation entry point to obtain a
plan or a target-specific diagnostic.

See [lowering a module to the shared representation](../../semantics/transpile-architecture.md#4-lower-module-ir) and
[the score-preservation criterion for a translation](../../semantics/transpile-correctness/index.md#2-correctness-criterion).

The support tables the plan derivation reads: object shapes and bounds,
family sentinels, argument constraints, and the wire forms they resolve
to. The module is implementation and exports no public names; the
lowering entry point is [`Lower`][quivers.transpile.Lower].

::: quivers.transpile.lower
