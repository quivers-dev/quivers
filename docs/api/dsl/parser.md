# Parser

Parsing identifies [declarations and expressions in source text](../../guides/dsl-overview.md#grammar) and records them as a
tree. This can detect malformed syntax before any tensors are created. A syntactically
valid tree may still contain incompatible types, unresolved names, or unsupported
effects that [later checking](../../semantics/adequacy.md#2-direct-correspondences) rejects.

Parsing QVR source into AST nodes.

::: quivers.dsl.parser
