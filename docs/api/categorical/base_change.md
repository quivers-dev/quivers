# Base Change

[Changing the enrichment](../../guides/categorical.md#base-change) changes what each weight means while retaining the finite input
and output objects. Including Boolean values in fuzzy scores preserves the entries 0 and
1; thresholding fuzzy scores discards distinctions between them. A conversion is a
functorial base change only when it [preserves the operations required by the composition
law](../../semantics/algebras.md#3-base-change) [@maclane1998categories].

The `BaseChange` interface and the `BoolToFuzzy` and `FuzzyToBool`
changes of enrichment.

::: quivers.categorical.base_change
