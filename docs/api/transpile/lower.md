# Lowering support

These are the support tables that [lowering](../../semantics/transpile-architecture.md#4-lower-module-ir)
reads to derive a plan: object shapes and bounds, family sentinels, argument constraints, and the
wire forms they resolve to. The module exports no public names, and the lowering entry point is
[`Lower`][quivers.transpile.Lower].

::: quivers.transpile.lower
