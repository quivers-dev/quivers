# Compiler

`Compiler` compiles a parsed QVR module to an executable `Program`
([compilation pipeline](../../guides/dsl-overview.md#compilation-pipeline)). It resolves names,
checks that domains and codomains agree, builds the runtime morphisms
([correspondences](../../semantics/adequacy.md#2-direct-correspondences)), and attaches the module's
checked [Quivers Indexed Effect Core](../qiec/index.md) (QIEC) form. The resulting program can be run or passed to an inference procedure, and
`CompileError` reports a rejected module.

For interactive inspection, `compile_env(defer_unresolved=True)` compiles declarations
whose dependencies have known finite sizes and retains the remaining declarations
with their authored types. `deferred_statements` lists this dependency closure in
source order. The compiler preserves the complete checked QIEC module; unresolved
sizes do not hide errors in independent declarations. Ordinary compilation still
requires evidence for each inferred size.

::: quivers.dsl.compiler
