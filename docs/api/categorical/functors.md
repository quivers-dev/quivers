# Functors

A functor maps objects and arrows while preserving identities and composition [@maclane1998categories]. [Mapping a
function over a list](../../guides/categorical.md#functors) gives a familiar example: mapping f and then g produces the same
list as mapping their composite. An implementation of the interfaces below must satisfy
the corresponding equations for its objects and arrows.

The `Functor` interface and its identity, composition, and free-monoid
implementations.

::: quivers.categorical.functors
    options:
      filters:
        - "!^_"
        - "!^[A-Z][A-Z0-9_]*$"

## Package constants

The singletons of `quivers.categorical`.

::: quivers.categorical
    options:
      members:
        - IDENTITY
        - EMPTY
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
