# Monad ↔ Arrow Bridges

The bridges translate between the monad and [arrow](https://en.wikipedia.org/wiki/Arrow_(computer_science))
interfaces [@hughes2000arrows]. `Kleisli` presents functions that return monadic values as
[Kleisli arrows](https://ncatlab.org/nlab/show/Kleisli+category), and `ArrowMonad` gives a monad
interface to an arrow that supports application: the identity monad for the function arrow and the
monad `m` for `Kleisli(m)`, while any other arrow raises `TypeError`. The `kleisli` and `arrow_monad` helpers build them
([bridges between the hierarchies](../../guides/effects.md#bridges-between-the-two-towers)), and
each construction is lawful only when the identity and composition laws of its argument hold
[@moggi1991notions; @wadler1992essence].

::: quivers.monadic.bridges
