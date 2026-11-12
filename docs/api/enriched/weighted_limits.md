# Weighted Limits

A [weighted limit](https://ncatlab.org/nlab/show/weighted+limit) of a diagram describes the
compatible ways of mapping into it, with a weight saying how much each object of the diagram
contributes [@kelly1982basic]. For a family of parallel morphisms $f_j : A \to B$ indexed by a
finite set $J$ and a weight $W$, the weighted limit is the pointwise meet
$\bigwedge_j [W(j), f_j]$ of residua and the weighted colimit the pointwise join
$\bigvee_j W(j) \otimes f_j$. `weighted_limit_morphisms` and `weighted_colimit_morphisms`
compute them when the `Algebra` is an `IdempotentSemiring`, so that its values form a complete
lattice, and the results then have the universal property
([weighted limits](../../guides/enriched.md#weighted-limits)).

::: quivers.enriched.weighted_limits
