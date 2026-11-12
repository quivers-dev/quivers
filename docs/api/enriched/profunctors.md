# Profunctors

A [profunctor](https://ncatlab.org/nlab/show/profunctor) relates objects of two categories without
choosing one output for each input [@kelly1982basic]. Its finite form here is a
[weighted table with source and target axes](../../guides/enriched.md#profunctors), and composing two
such tables [matches their common axis and reduces over it](../../semantics/setting.md#2-v-enriched-relations).
The general definition also requires the source and target categories to act compatibly on the
table.

::: quivers.enriched.profunctors
