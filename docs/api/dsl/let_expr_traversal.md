# Let-expression traversal

Walking an expression requires knowing which occurrences of a name are [bound locally](../../reference/qvr/types-and-indexed-families.md#pure-expressions).
These helpers find free names and [substitute expressions without capturing a variable](../qiec/substitution.md)
accidentally. For instance, replacing an outer `x` must not replace the unrelated `x`
introduced by an inner lambda.

Binder-aware walks, free-name analysis, and substitution over the
let-expression AST.

::: quivers.dsl.let_expr_traversal
