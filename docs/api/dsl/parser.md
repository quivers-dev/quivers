# Parser

Parsing identifies declarations and expressions in source text and records them as a
tree. This can detect malformed syntax before any tensors are created. A syntactically
valid tree may still contain incompatible types, unresolved names, or unsupported
effects that later checking rejects.

See [the source grammar](../../guides/dsl-overview.md#grammar) and
[the correspondence between syntax and runtime objects](../../semantics/adequacy.md#2-direct-correspondences).

The [Python AST documentation](https://docs.python.org/3/library/ast.html) illustrates the distinction between source text, syntax trees, and compilation in another language.

Parsing QVR source into AST nodes.

::: quivers.dsl.parser
