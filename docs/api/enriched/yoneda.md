# Yoneda

A representable presheaf records all the maps into an object. The [Yoneda lemma](../../guides/enriched.md#yoneda) says that
a natural transformation from this presheaf is determined by its value at the object’s
identity map [@kelly1982basic]. The implementations below use finite tensors to represent these maps and
evaluate the corresponding equations.

Yoneda embeddings and the Yoneda lemma in enriched categories.

::: quivers.enriched.yoneda
