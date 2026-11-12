# Weighted Limits

A [weighted limit](https://ncatlab.org/nlab/show/weighted+limit) of a diagram describes the
compatible ways of mapping into it, with a weight saying how much each object of the diagram
contributes [@kelly1982basic]. `weighted_limit`, `weighted_colimit`, and their variants on
morphisms compute finite tensors with the chosen `Algebra`'s operations and its internal hom.
Whether a result has the general universal property depends on the diagram and the weight
([weighted limits](../../guides/enriched.md#weighted-limits)).

::: quivers.enriched.weighted_limits
