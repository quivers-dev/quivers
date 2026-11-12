# Yoneda

A representable presheaf records every map into one object. The
[Yoneda lemma](https://ncatlab.org/nlab/show/Yoneda+lemma) says that a natural transformation out of
it is determined by its value at that object's identity map [@kelly1982basic]
([Yoneda](../../guides/enriched.md#yoneda)). The module represents these maps as finite tensors and
evaluates the corresponding equations.

::: quivers.enriched.yoneda
