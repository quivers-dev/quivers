# Comonads

A comonad describes computations that extract a value from a context and extend
computations over that context. Its counit laws constrain how extraction and duplication
interact. The `DiagonalComonad` documentation below gives a pair example where the
implemented operations fail the right counit law.

See [extracting and extending context](../../guides/monadic.md#comonads) and
[the pair example that exposes a counit-law failure](../../guides/monadic.md#diagonal-comonad).

For the semantics of monadic computation, see [@moggi1991notions; @wadler1992essence].

Comonad definitions with counits, comultiplications, and comonad laws.

::: quivers.monadic.comonads
