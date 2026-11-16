# Types

Static terms are the types, indices, shapes, and effect references of the core
([family declarations](../../reference/qvr/types-and-indexed-families.md#family-declarations)). A
[vector's length index](../../tutorials/qvr/09-indexed-data.md#2-fold-without-a-shape-check), for
instance, constrains how many elements a value has, and the checker tracks that constraint through
constructor applications and effectful computations. The module also renders static terms as text.

`IndexSum` represents canonical addition of Nat variables and a nonnegative
constant. `add_indices` folds constants and nested sums; `shift_index`
adds an offset or removes one already proved present in the constant part.
Rigid identities survive both operations. The checked source form is
`nat_add(n, 1)`, described in the
[index reference](../../reference/qvr/types-and-indexed-families.md#natural-number-extent-addition).

`solve_index_equation` cancels affine terms only when the solution remains
a representable nonnegative Nat expression. `infer_index_extents` solves
exact shape equations, reports inconsistent or nonintegral evidence, and
leaves underdetermined variables unbound. Neither helper guesses that
unrelated axes coincide because their extents happen to agree.

::: quivers.qiec.types

::: quivers.qiec.index_arithmetic
