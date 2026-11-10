# Overview

The domain-specific language (DSL) is the `.qvr` notation used to declare and combine
models. `parse` turns text into a syntax tree; `load` and `loads` also [compile it into
an executable program](../../guides/dsl-overview.md#compilation-pipeline). Start with [those entry points](../../tutorials/qvr/01-first-model.md#compile-and-fit) to use QVR, and consult the
lower-level modules when extending the language or its tooling.

Parsing, compiling, and printing QVR source, with the public entry
points: `load` and `loads` compile source to a `Program`, `parse`
produces the AST that `Compiler` lowers and `module_to_source` prints,
and the QIEC lowering checks a module against the kernel. The package
re-exports the public names of its modules; the AST node classes live
in [`quivers.dsl.ast_nodes`](ast_nodes.md).

::: quivers.dsl
