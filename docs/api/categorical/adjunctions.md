# Adjunctions

An [adjunction](https://ncatlab.org/nlab/show/adjunction) pairs two functors so that maps out of
the first correspond to maps into the second [@maclane1998categories]. A function from a set of
generators into a monoid, for instance, determines a unique monoid homomorphism from the
[free monoid](../../guides/core.md#freemonoid) on those generators. The
[unit and counit](../../guides/categorical.md#adjunctions) express this correspondence, and the
adjunction is an equivalence when both are isomorphisms. The module gives the `Adjunction`
interface, with its unit and counit and checks of the triangle identities.

::: quivers.categorical.adjunctions
