# Comonads

A comonad describes computations that [extract a value from a context and extend
computations over that context](../../guides/monadic.md#comonads). Its counit laws constrain how extraction and duplication
interact. `CofreeComonad` is the product (environment) comonad `A × S`.

Comonad definitions with counits, comultiplications, and comonad laws.

::: quivers.monadic.comonads
