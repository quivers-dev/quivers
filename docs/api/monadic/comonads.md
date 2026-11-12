# Comonads

A [comonad](https://ncatlab.org/nlab/show/comonad) is the dual of a monad: a counit extracts a
value from its context, and a comultiplication duplicates the context so that computations extend
over it [@maclane1998categories]. The counit laws constrain how extraction and duplication
interact ([comonads](../../guides/monadic.md#comonads)). The module gives the `Comonad` interface,
its co-Kleisli category, and `CofreeComonad`. `DiagonalComonad` exposes pair operations, the first
projection and the diagonal, that fail the right counit law whenever the two components of a pair
differ ([pair example](../../guides/monadic.md#diagonal-comonad)).

::: quivers.monadic.comonads
