# Monad Transformers

A monad transformer adds an effect to an existing monad [@moggi1991notions; @wadler1992essence]. Transformer order determines
how [effects interact](../../guides/effects.md#stdlib-effect-instances): combining state with failure can retain or discard a state change
when a computation fails. Choose the order according to the behavior required at that
failure point.

`StateT`, `ReaderT`, `MaybeT`, `ContT`, and `WriterT` stack one effect
on another.

::: quivers.monadic.transformers
