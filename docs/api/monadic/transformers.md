# Monad Transformers

A monad transformer adds an effect to an existing monad. Transformer order determines
how effects interact: combining state with failure can retain or discard a state change
when a computation fails. Choose the order according to the behavior required at that
failure point.

See [the effects that a transformer combines](../../guides/effects.md#stdlib-effect-instances) and
[the mathematical interpretation of the interfaces](../../semantics/effects.md#2-typeclass-denotations).

For the semantics of monadic computation, see [@moggi1991notions; @wadler1992essence].

`StateT`, `ReaderT`, `MaybeT`, `ContT`, and `WriterT` stack one effect
on another.

::: quivers.monadic.transformers
