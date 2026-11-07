# Adjunctions

An adjunction relates two ways of specifying the same map [@maclane1998categories]. A function from a set of
generators into a monoid determines a unique monoid homomorphism from the [free monoid](../../guides/core.md#freemonoid) on
those generators. The [unit and counit](../../guides/categorical.md#adjunctions) express this correspondence; an equivalence
additionally requires both to be isomorphisms.

The `Adjunction` interface, the `FreeForgetfulAdjunction` implementation,
and their unit and counit operations.

::: quivers.categorical.adjunctions
