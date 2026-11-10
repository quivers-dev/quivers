# Lowering support

[Lowering uses shape, support, and family information](../../semantics/transpile-architecture.md#4-lower-module-ir) to express a model in the
representation consumed by target renderers. These support tables guide that process.
They are implementation details; use the public transpilation entry point to obtain a
plan or a target-specific diagnostic.

The support tables the plan derivation reads: object shapes and bounds,
family sentinels, argument constraints, and the wire forms they resolve
to. The module is implementation and exports no public names; the
lowering entry point is [`Lower`][quivers.transpile.Lower].

::: quivers.transpile.lower
