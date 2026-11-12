# Objects

An object names [what a tensor axis ranges over](../../getting-started/concepts.md#1-an-object-names-the-possible-values),
so a set of three users and a set of three items have the same size and different meanings. The
module defines finite and enumerated sets, [products and coproducts](../../guides/core.md#productset-and-coproductset)
(tuples and tagged alternatives) [@maclane1998categories], free monoids, and `FreeResiduated`, the
categorial (Lambek) types built from a set of atoms with `/`, `\`, and `*` up to a fixed depth
[@lambek1958mathematics]. These objects index finite axes. Continuous supports have their own
space types.

::: quivers.core.objects
    options:
      filters:
        - "!^_"
        - "!^Unit$"

## Package constants

The constants and type aliases of `quivers.core`: the terminal object,
the numerical margin, the algebra and homomorphism singletons, the
homomorphism registry, and the transformation alias.

::: quivers.core
    options:
      members:
        - Unit
        - EPS
        - PRODUCT_FUZZY
        - BOOLEAN
        - LUKASIEWICZ
        - GODEL
        - TROPICAL
        - MAX_PLUS
        - LOG_PROB
        - REAL
        - PROBABILITY
        - COUNTING
        - MARKOV
        - REICHENBACH
        - BOOLEAN_DUAL
        - DUAL_LUKASIEWICZ
        - DUAL_GODEL
        - EXPECTATION
        - LOG_PROB_HOM
        - MAX_PLUS_HOM
        - MATERIAL_IMPLICATION
        - PROBABILITY_CLAMP
        - COUNTING_FROM_REAL
        - PROBABILITY_TO_REAL
        - COUNTING_TO_REAL
        - HOMOMORPHISM_REGISTRY
        - TransValue
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
