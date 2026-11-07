# Algebraic Effects

An algebraic-effect program records requests separately from their interpretation. A
[handler supplies a result for each request](../../guides/effects.md#algebraic-effects-handlers) and decides how computation continues [@plotkin2009handlers; @bauer2015programming]. This
permits the same request structure to receive [different interpretations](../../semantics/effects.md#2-typeclass-denotations). These Python
interfaces describe explicit operation trees; source-level lexical effect instances are
documented separately in the QVR reference.

Plotkin-style effect signatures and handler-based interpretation via
`Operation`, `EffectSignature`, `Handler`, and `FreeMonad`.

::: quivers.monadic.algebraic
