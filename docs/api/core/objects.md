# Objects

Use an object to say [what each tensor axis ranges over](../../getting-started/concepts.md#1-an-object-names-the-possible-values). A set of three users and a set
of three items have the same size but different meanings. [Products and coproducts](../../guides/core.md#productset-and-coproductset)
contain tuples and tagged alternatives, respectively [@maclane1998categories].
These objects describe finite indexing; continuous supports have their own
space types.

Categorical objects including finite and enumerated sets, products,
coproducts, free monoids, and free residuated universes.

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
        - LOG_PROB_MAP
        - MAX_PLUS_MAP
        - MATERIAL_IMPLICATION
        - PROBABILITY_CLAMP
        - COUNTING_FROM_REAL
        - PROBABILITY_TO_REAL
        - COUNTING_TO_REAL
        - CARRIER_MAP_REGISTRY
        - TransValue
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
