# Plan

`Lower` derives a program's target plan from its checked QIEC module, and every target renders from
that plan ([lowering](../../semantics/transpile-architecture.md#4-lower-module-ir)). The plan collects
the declarations and dependencies a target needs, checks support before any source is emitted, and
names the operation responsible when a target refuses a model.

::: quivers.transpile.plan
