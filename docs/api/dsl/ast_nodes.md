# AST Nodes

An [abstract syntax tree (AST)](https://docs.python.org/3/library/ast.html) represents source structure as records. A declaration
contains its name, arguments, and expressions, giving tools access to these parts during
analysis or transformation. [Elaboration then resolves names](../../guides/dsl-overview.md#compilation-pipeline) and [makes implicit
operations explicit](../../semantics/adequacy.md#2-direct-correspondences).

Abstract syntax tree node definitions for the QVR DSL.

::: quivers.dsl.ast_nodes
