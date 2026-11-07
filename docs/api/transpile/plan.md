# Plan

A [transpilation plan](../../semantics/transpile-architecture.md#4-lower-module-ir) collects the declarations and dependencies required by a target. It
checks support before emitting source and identifies the operation responsible when
translation is refused. Inspect the plan to determine which part of a model needs a
different target or representation.

Deriving a program's target plan from its checked kernel module, and the
lowering entry point every target renders from.

::: quivers.transpile.plan
