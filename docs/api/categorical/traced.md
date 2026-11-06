# Traced Monoidal

A categorical trace connects an output component to a matching input component. For
finite tensors, this amounts to identifying the two axes and reducing the resulting
diagonal. Use the stated contraction rule to determine which values combine and which
axes remain.

See [the finite tensor trace](../../guides/categorical.md#traced-monoidal-structure) and
[the table interpretation of objects and arrows](../../tutorials/qvr/07-categorical.md#begin-with-composable-tables).

For the categorical definitions and their laws, see [@maclane1998categories].

Cartesian and iterative traced monoidal structures, with `trace` and
`partial_trace` operations.

::: quivers.categorical.traced
