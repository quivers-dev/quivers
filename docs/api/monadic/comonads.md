# Comonads

A [comonad](https://ncatlab.org/nlab/show/comonad) is the dual of a monad: a counit extracts a
value from its context, and a comultiplication duplicates the context so that computations extend
over it [@maclane1998categories]. The counit laws constrain how extraction and duplication
interact ([comonads](../../guides/monadic.md#comonads)). The module gives the `Comonad` interface,
its co-Kleisli category, and `CofreeComonad`, the product (environment) comonad $A \times S$.

::: quivers.monadic.comonads
