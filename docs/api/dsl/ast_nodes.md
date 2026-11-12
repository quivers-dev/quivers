# AST Nodes

The AST node classes of QVR, each a didactic `Model`, record the structure of a parsed module: a
declaration holds its name, arguments, and expressions, so tools can read and transform these
parts. [Elaboration](../../guides/dsl-overview.md#compilation-pipeline), the compiler step that
rewrites declarations into explicit core terms, then resolves names and makes implicit operations
explicit ([correspondences](../../semantics/adequacy.md#2-direct-correspondences)).

::: quivers.dsl.ast_nodes
