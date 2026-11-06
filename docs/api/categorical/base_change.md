# Base Change

Changing the enrichment changes what each weight means while retaining the finite input
and output objects. Including Boolean values in fuzzy scores preserves the entries 0 and
1; thresholding fuzzy scores discards distinctions between them. A conversion is a
functorial base change only when it preserves the operations required by the composition
law.

See [converting weights between algebras](../../guides/categorical.md#base-change) and
[the conditions for preserving composition](../../semantics/algebras.md#3-base-change).

For the categorical definitions and their laws, see [@maclane1998categories].

The `BaseChange` interface and the `BoolToFuzzy` and `FuzzyToBool`
changes of enrichment.

::: quivers.categorical.base_change
