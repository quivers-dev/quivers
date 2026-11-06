# Profunctors

A profunctor can relate objects on two sides without choosing a single output for each
input. Here its finite representation is a weighted table with explicit source and
target axes. Composing two such tables matches their common axis and combines
alternatives. The general definition additionally requires compatible actions of the
source and target categories.

For the background on weighted relationships, see Milewski's [enriched categories chapter](https://bartoszmilewski.com/2017/05/13/enriched-categories/) [@milewski2017enriched].

See [profunctors as weighted tables](../../guides/enriched.md#profunctors) and
[the weighted relations used by these constructions](../../semantics/setting.md#2-v-enriched-relations).

For the general enriched constructions behind this finite interface, see [@kelly1982basic].

Profunctors (bimodules) in enriched categories.

::: quivers.enriched.profunctors
