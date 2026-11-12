# Monad Transformers

A monad transformer adds one effect to an existing monad [@moggi1991notions; @wadler1992essence].
`StateT`, `ReaderT`, `MaybeT`, `ContT`, and `WriterT` each stack their effect on another. The order
of a stack determines how the [effects interact](../../guides/effects.md#stdlib-effect-instances):
`StateT` over `Maybe` discards the state when a computation fails, while `MaybeT` over `State`
keeps it.

::: quivers.monadic.transformers
