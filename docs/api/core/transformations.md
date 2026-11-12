# Transformations

A [transformation](../../guides/transformations.md#first-class-transformations) rewrites a morphism's
entries into another `Algebra`. An entrywise map (`AlgebraHomomorphism`), such as a logarithm, acts
on each entry alone. A `MorphismTransformation`, such as softmax or L1 row normalization, reads a
whole row. `compose_trans` builds sequences of both, and `Morphism.change_base` applies them. An
entrywise map that preserves $\otimes$, $\oplus$, and their units commutes with composition, and
then applying it is a [change of base](https://ncatlab.org/nlab/show/change+of+enriching+category)
[@kelly1982basic].

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
