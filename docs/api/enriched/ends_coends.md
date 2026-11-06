# Ends and Coends

For a tensor with two occurrences of the same finite index, these routines extract the
diagonal and reduce it. The `coend` routine uses the algebra’s join, and `end` uses its
meet. In the general categorical construction, an end or coend must also satisfy
compatibility equations for the indexing category’s morphisms; the guide states the
scope of the finite implementation.

See [the finite implementation of ends and coends](../../guides/enriched.md#ends-and-coends) and
[the weighted relations used by these constructions](../../semantics/setting.md#2-v-enriched-relations).

For the general enriched constructions behind this finite interface, see [@kelly1982basic].

End and coend computations in enriched categories.

::: quivers.enriched.ends_coends
