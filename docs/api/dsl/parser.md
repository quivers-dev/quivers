# Parser

`parse` and `parse_file` read QVR source into AST nodes
([grammar](../../guides/dsl-overview.md#grammar)) and raise `ParseError` on malformed syntax, before
any tensor is created. A tree that parses may still have mismatched types, unresolved names, or
unsupported effects, which [later checking](../../semantics/adequacy.md#2-direct-correspondences)
rejects.

::: quivers.dsl.parser
