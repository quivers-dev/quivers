# Plan

A transpilation plan collects the declarations and dependencies required by a target. It
checks support before emitting source and identifies the operation responsible when
translation is refused. Inspect the plan to determine which part of a model needs a
different target or representation.

See [deriving the declarations required by a target](../../semantics/transpile-architecture.md#4-lower-module-ir) and
[the score-preservation criterion for a translation](../../semantics/transpile-correctness/index.md#2-correctness-criterion).

Deriving a program's target plan from its checked kernel module, and the
lowering entry point every target renders from.

::: quivers.transpile.plan
