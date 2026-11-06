# Yoneda

A representable presheaf records all the maps into an object. The Yoneda lemma says that
a natural transformation from this presheaf is determined by its value at the object’s
identity map. The implementations below use finite tensors to represent these maps and
evaluate the corresponding equations.

Milewski's [Yoneda lemma chapter](https://bartoszmilewski.com/2015/09/01/the-yoneda-lemma/) develops the identity-map argument through programming examples [@milewski2015yoneda].

See [the Yoneda construction](../../guides/enriched.md#yoneda) and
[the weighted relations used by these constructions](../../semantics/setting.md#2-v-enriched-relations).

For the general enriched constructions behind this finite interface, see [@kelly1982basic].

Yoneda embeddings and the Yoneda lemma in enriched categories.

::: quivers.enriched.yoneda
