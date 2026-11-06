# Kan Extensions

Suppose detailed states map to coarser groups. For each group, collect the original
states that map to it; this collection is its fiber. The implemented left extension
joins weights over that fiber, while the right extension meets them. A projection that
forgets one coordinate thus turns this operation into reduction over the forgotten
coordinate.

See [the finite implementation of Kan extensions](../../guides/enriched.md#kan-extensions) and
[the weighted relations used by these constructions](../../semantics/setting.md#2-v-enriched-relations).

For the general enriched constructions behind this finite interface, see [@kelly1982basic].

Left and right Kan extensions in enriched categories.

::: quivers.enriched.kan_extensions
