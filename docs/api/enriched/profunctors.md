# Profunctors

A profunctor can relate objects on two sides without choosing a single output for each
input. Here its finite representation is a [weighted table with explicit source and
target axes](../../guides/enriched.md#profunctors). Composing two such tables [matches their common axis and combines
alternatives](../../semantics/setting.md#2-v-enriched-relations). The general definition additionally requires compatible actions of the
source and target categories [@kelly1982basic].

Profunctors (bimodules) in enriched categories.

::: quivers.enriched.profunctors
