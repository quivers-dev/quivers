# Monad ↔ Arrow Bridges

The bridges translate between monadic and arrow interfaces. Kleisli arrows expose
computations that return monadic values, while suitable arrows with application support
can provide a monadic interface. These constructions require the corresponding identity
and composition laws.

Milewski's [Kleisli categories chapter](https://bartoszmilewski.com/2014/12/23/kleisli-categories/) motivates the change in composition using functions that return additional information [@milewski2014kleisli].

See [the Kleisli and arrow bridges](../../guides/effects.md#bridges-between-the-two-towers) and
[the mathematical interpretation of the interfaces](../../semantics/effects.md#2-typeclass-denotations).

For the semantics of monadic computation, see [@moggi1991notions; @wadler1992essence].

`Kleisli` and `ArrowMonad`, together with the `kleisli` and
`arrow_monad` helpers that connect the two typeclass hierarchies.

Sources for the constructions below: [@hughes2000arrows].

::: quivers.monadic.bridges
