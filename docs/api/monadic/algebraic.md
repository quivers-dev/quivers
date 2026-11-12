# Algebraic Effects

An [algebraic effect](https://en.wikipedia.org/wiki/Effect_system) program records its requests
separately from their interpretation, and a
[handler](../../guides/effects.md#algebraic-effects-handlers) answers each request and decides how
the computation continues [@plotkin2003algebraic; @plotkin2009handlers; @bauer2015programming]. One
request structure can thus receive [different interpretations](../../semantics/effects.md#2-typeclass-denotations).
`Operation`, `EffectSignature`, `Handler`, and `FreeMonad` build and interpret explicit trees of
operations in Python. The effect instances of QVR source are specified in the
[QVR reference](../../reference/qvr/effects-and-handlers.md).

::: quivers.monadic.algebraic
