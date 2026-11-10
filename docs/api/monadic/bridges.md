# Monad ↔ Arrow Bridges

The bridges translate between monadic and arrow interfaces [@hughes2000arrows]. [Kleisli arrows](../../guides/effects.md#bridges-between-the-two-towers) expose
computations that return monadic values, while suitable arrows with application support
can provide a monadic interface. These constructions require the corresponding identity
and composition laws [@moggi1991notions; @wadler1992essence].

`Kleisli` and `ArrowMonad`, together with the `kleisli` and
`arrow_monad` helpers that connect the two typeclass hierarchies.

::: quivers.monadic.bridges
