# Compiler

`Compiler` compiles a parsed QVR module to an executable `Program`
([compilation pipeline](../../guides/dsl-overview.md#compilation-pipeline)). It resolves names,
checks that domains and codomains agree, builds the runtime morphisms
([correspondences](../../semantics/adequacy.md#2-direct-correspondences)), and attaches the module's
checked [Quivers Indexed Effect Core](../qiec/index.md) (QIEC) form. The resulting program can be run or passed to an inference procedure, and
`CompileError` reports a rejected module.

::: quivers.dsl.compiler
