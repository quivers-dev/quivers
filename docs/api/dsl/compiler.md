# Compiler

The [compiler parses declarations, resolves names, checks compatible interfaces](../../guides/dsl-overview.md#compilation-pipeline), and
[constructs runtime objects](../../semantics/adequacy.md#2-direct-correspondences). Use this API when loading a QVR module from Python or
extending the compilation process. The resulting program can then be executed or passed
to an inference procedure.

Lowering from a DSL AST to an executable `Program`.

::: quivers.dsl.compiler
