# Let-expression traversal

These functions walk the let-expression AST with attention to binders: they find the free names of
an expression ([pure expressions](../../reference/qvr/types-and-indexed-families.md#pure-expressions))
and substitute without [capturing a variable](../qiec/substitution.md). Replacing an outer `x`, for
instance, must not replace an unrelated `x` bound by an inner lambda.

`LetExprExtension` lets a downstream IR expose additional binder-free nodes through the same
operations. Its adapter supplies children and reconstruction; canonical traversal retains control
of scope and capture avoidance. `substitute_let_expr` can also use a separate replacement in direct
index positions, so a target's index origin does not require another substitution implementation.

::: quivers.dsl.let_expr_traversal
