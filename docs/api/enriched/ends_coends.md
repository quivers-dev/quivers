# Ends and Coends

For a tensor with two occurrences of the same finite index, these routines extract the
diagonal and reduce it. The `coend` routine uses the algebra’s join, and `end` uses its
meet. In the general categorical construction, an end or coend must also satisfy
compatibility equations for the indexing category’s morphisms [@kelly1982basic].
The guide states the
[scope of the finite implementation](../../guides/enriched.md#ends-and-coends).

End and coend computations in enriched categories.

::: quivers.enriched.ends_coends
