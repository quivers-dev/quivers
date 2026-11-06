# Functors

A functor maps objects and arrows while preserving identities and composition. Mapping a
function over a list gives a familiar example: mapping f and then g produces the same
list as mapping their composite. An implementation of the interfaces below must satisfy
the corresponding equations for its objects and arrows.

Milewski develops mapping over containers in [his chapter on functors](https://bartoszmilewski.com/2015/01/20/functors/#:~:text=The%20List%20Functor) [@milewski2015functors].

See [the list-mapping example and functor laws](../../guides/categorical.md#functors) and
[the table interpretation of objects and arrows](../../tutorials/qvr/07-categorical.md#begin-with-composable-tables).

For the categorical definitions and their laws, see [@maclane1998categories].

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
