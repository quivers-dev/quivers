# Comonads

A comonad describes computations that [extract a value from a context and extend
computations over that context](../../guides/monadic.md#comonads). Its counit laws constrain how extraction and duplication
interact. The `DiagonalComonad` documentation below gives a [pair example](../../guides/monadic.md#diagonal-comonad) where the
implemented operations fail the right counit law.

Comonad definitions with counits, comultiplications, and comonad laws.

::: quivers.monadic.comonads
