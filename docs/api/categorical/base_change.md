# Change of Base

[Change of base](../../guides/categorical.md#base-change) reinterprets each weight in another
`Algebra` and keeps the finite input and output objects. `BoolToFuzzy` includes Boolean weights
among fuzzy scores and keeps the entries 0 and 1. `FuzzyToBool` thresholds fuzzy scores and
discards the distinctions between them. A conversion commutes with composition only when it
[preserves the operations composition uses](../../semantics/algebras.md#3-base-change)
[@kelly1982basic]. The module gives the `BaseChange` interface and these two instances.

::: quivers.categorical.base_change
