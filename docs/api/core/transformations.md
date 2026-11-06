# Transformations

Transformations change a table’s entries or their interpretation. A logarithm acts on
each entry; softmax uses an entire row to produce normalized weights. The source and
target algebras specify the available operations before and after conversion. Preserving
composition imposes additional equations on the transformation.

See [applying transformations](../../guides/transformations.md#first-class-transformations) and
[a Bayesian inversion example](../../tutorials/python/06-first-class-trans.md#bayes-inversion).

For change of enrichment and its preservation requirements, see [@kelly1982basic].

Change-of-base transformations between algebras: pointwise algebra
homomorphisms, shape-aware morphism transformations, and the sequences
`compose_trans` builds from them. Each is consumed by
`Morphism.change_base`.

::: quivers.core.algebra_morphisms
    options:
      filters:
        - "!^_"
        - "!^[A-Z][A-Z0-9_]*$"

::: quivers.core.morphism_transformations

::: quivers.core.trans
    options:
      filters:
        - "!^_"
        - "!^TransValue$"

The homomorphism singletons, their registry, and the `TransValue`
alias are documented on the [core overview](objects.md#package-constants).
