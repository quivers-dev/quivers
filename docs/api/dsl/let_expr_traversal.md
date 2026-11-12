# Let-expression traversal

These functions walk the let-expression AST with attention to binders: they find the free names of
an expression ([pure expressions](../../reference/qvr/types-and-indexed-families.md#pure-expressions))
and substitute without [capturing a variable](../qiec/substitution.md). Replacing an outer `x`, for
instance, must not replace an unrelated `x` bound by an inner lambda.

::: quivers.dsl.let_expr_traversal
