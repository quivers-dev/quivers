# Let-expression traversal

Walking an expression requires knowing which occurrences of a name are bound locally.
These helpers find free names and substitute expressions without capturing a variable
accidentally. For instance, replacing an outer `x` must not replace the unrelated `x`
introduced by an inner lambda.

See [bound names in pure expressions](../../reference/qvr/types-and-indexed-families.md#pure-expressions) and
[substitution that respects variable binding](../qiec/substitution.md).

Binder-aware walks, free-name analysis, and substitution over the
let-expression AST.

::: quivers.dsl.let_expr_traversal
