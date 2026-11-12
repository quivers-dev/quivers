# Functors

A [functor](https://ncatlab.org/nlab/show/functor) maps objects and arrows while preserving
identities and composition [@maclane1998categories]. [Mapping a function over a list](../../guides/categorical.md#functors)
is the familiar case: mapping $f$ and then $g$ gives the same list as mapping their composite. The
module gives the `Functor` interface and its identity, composite, and free monoid instances, and an
implementation must satisfy the functor equations for its objects and arrows.

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
