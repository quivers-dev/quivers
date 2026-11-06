# Adjunctions

An adjunction relates two ways of specifying the same map. A function from a set of
generators into a monoid determines a unique monoid homomorphism from the free monoid on
those generators. The unit and counit express this correspondence; an equivalence
additionally requires both to be isomorphisms.

Milewski's [adjunctions chapter, especially “Adjunctions and Hom-Sets”](https://bartoszmilewski.com/2016/04/18/adjunctions/#:~:text=Adjunctions%20and%20Hom-Sets), explains the correspondence between maps [@milewski2016adjunctions].

See [the free/forgetful adjunction example](../../guides/categorical.md#adjunctions) and
[the free monoid constructor](../../guides/core.md#freemonoid).

For the categorical definitions and their laws, see [@maclane1998categories].

The `Adjunction` interface, the `FreeForgetfulAdjunction` implementation,
and their unit and counit operations.

::: quivers.categorical.adjunctions
