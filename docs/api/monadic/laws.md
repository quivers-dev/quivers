# Typeclass Laws

`check_functor_laws` and `check_monad_laws` test the
[identity and associativity laws](../../guides/monadic.md#monads) of a type class instance on
chosen objects [@moggi1991notions; @wadler1992essence]. A passing check covers the values tested,
and a proof that a law holds must cover the implementation's whole domain.

::: quivers.monadic.laws
