# Compiler

The compiler parses declarations, resolves names, checks compatible interfaces, and
constructs runtime objects. Use this API when loading a QVR module from Python or
extending the compilation process. The resulting program can then be executed or passed
to an inference procedure.

See [the stages of compilation](../../guides/dsl-overview.md#compilation-pipeline) and
[the correspondence between syntax and runtime objects](../../semantics/adequacy.md#2-direct-correspondences).

Lowering from a DSL AST to an executable `Program`.

::: quivers.dsl.compiler
