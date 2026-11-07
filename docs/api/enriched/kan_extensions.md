# Kan Extensions

Suppose detailed states map to coarser groups. For each group, collect the original
states that map to it; this collection is its fiber. The implemented [left extension](../../guides/enriched.md#kan-extensions)
joins weights over that fiber, while the right extension meets them. A projection that
forgets one coordinate thus turns this operation into reduction over the forgotten
coordinate [@kelly1982basic].

Left and right Kan extensions in enriched categories.

::: quivers.enriched.kan_extensions
