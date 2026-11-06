# AST Nodes

An abstract syntax tree (AST) represents source structure as records. A declaration
contains its name, arguments, and expressions, giving tools access to these parts during
analysis or transformation. Elaboration then resolves names and makes implicit
operations explicit.

See [the stages from source text to a program](../../guides/dsl-overview.md#compilation-pipeline) and
[the correspondence between syntax and runtime objects](../../semantics/adequacy.md#2-direct-correspondences).

The [Python AST documentation](https://docs.python.org/3/library/ast.html) provides a familiar example of representing syntax as typed tree nodes; QVR uses its own node vocabulary.

Abstract syntax tree node definitions for the QVR DSL.

::: quivers.dsl.ast_nodes
