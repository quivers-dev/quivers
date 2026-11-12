# `quivers.dsl`

QVR is the modeling language of quivers, written in `.qvr` files. `load` and `loads` parse, check,
and [compile](../../guides/dsl-overview.md#compilation-pipeline) source to a `Program`, `parse`
produces the AST that `Compiler` compiles and `module_to_source` prints, and the QIEC lowering
elaborates a module into the [Quivers Indexed Effect Core](../qiec/index.md) and checks it.
The package re-exports the public names of its modules, and the AST node classes are documented
in [`quivers.dsl.ast_nodes`](ast_nodes.md).

::: quivers.dsl
